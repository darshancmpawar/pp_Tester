"""
ML Core — dynamic model training + prediction for any client's feature schema.

The model architecture is built dynamically from the client's encoder_columns list.
All artifacts are prefixed with the client key (e.g. tekion_per_pax_tf_model.keras).
"""
import os, math, random
from collections import defaultdict
import numpy as np, pandas as pd, tensorflow as tf, joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Where trained model files and encoders are stored. Can be overridden via env
# var before importing this module (scripts/train.py does this).
ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    os.path.join(os.path.dirname(__file__), "artifacts"),
)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ═══ TUNING: training hyperparameters ═════════════════════════════════════════
# These apply to EVERY client — there is one architecture, shaped per client by
# its encoder_columns. The datasets are small (a few thousand rows each), which
# is what every value below is sized for. Change one, then retrain all clients
# (`python scripts/train.py --client all --force`) and compare the printed RMSE;
# a typical per-pax RMSE is a few hundredths of a kg.

SEED       = 42   # fixed so two runs on the same dataset give the same model
CTX_LEN    = 10   # max co-menu items kept as context per row. A full menu is
                  # 8–10 dishes, so 10 covers a day without padding waste.
EPOCHS     = 20   # enough to converge on a few-thousand-row set; more overfits
BATCH      = 32   # small batches — the datasets are too small for 128+
VAL_SPLIT  = 0.1  # fraction of TRAINING rows Keras holds out to watch val_loss
TEST_SIZE  = 0.2  # fraction of DATES held out as the final (reported) test set

# TUNING: embedding width per feature, in floats. Roughly "how much the model
# is allowed to memorise about this column". Wider = more capacity and more
# overfitting risk; the rule of thumb here is width ∝ cardinality:
#   menu_items    hundreds of dishes   → 8
#   sub_category  tens                 → 4
#   category      ~10                  → 4
#   weekday       5–7                  → 2
#   day_type      4, holiday_type 4, meal_day 2 → 2
# A column missing from this dict falls back to 2 (see the model builder).
EMB_DIMS = {
    "menu_items":   8,
    "sub_category": 4,
    "category":     4,
    "weekday":      2,
    "day_type":     2,
    "holiday_type": 2,
    "meal_day":     2,
}


# ── Text normalisation helpers ────────────────────────────────────────────────

def norm(s):
    return s.strip().lower() if isinstance(s, str) else s

def token(v):
    """Normalise any raw value to a safe categorical string token."""
    if pd.isna(v):
        return "unknown"
    t = norm(v) if isinstance(v, str) else norm(str(v))
    return t if t else "unknown"

def cats_match(a, b):
    return a is not None and b is not None and norm(a) == norm(b)

def floor005(v):
    """
    Round DOWN to the nearest 0.005 kg — the per-pax output granularity.

    Down, not nearest: the vendor bump in planner.py is what adds the safety
    margin, so the client-facing PP stays conservative. 5 g is the finest
    difference a serving spoon can hold; see the matching note in planner.py.
    """
    return math.floor(v / 0.005) * 0.005

def fmt_cols(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(
                lambda x: f"{x:.3f}".rstrip("0").rstrip(".") if isinstance(x, (int, float)) else x
            )
    return df


# ── Artifact path helper ──────────────────────────────────────────────────────

def _ap(ck, fn):
    """Return the full path for a client-prefixed artifact file."""
    return os.path.join(ARTIFACT_DIR, f"{ck}_{fn}")

def artifacts_exist(ck, encoder_columns):
    """True when every artifact file the app needs for this client is on disk."""
    expected = (
        ["per_pax_tf_model.keras"]
        + [f"{c}_encoder.pkl" for c in encoder_columns]
        + ["item_to_subcat.pkl", "item_to_cat.pkl", "cat_to_subs.pkl"]
    )
    return all(os.path.exists(_ap(ck, f)) for f in expected)


# ── In-process model cache ────────────────────────────────────────────────────
# Streamlit Cloud runs ONE Python process for all user sessions (threads).
# Loading a Keras model or encoder from disk on every prediction would be
# too slow. _model_cache holds already-loaded objects keyed by "{ck}:{type}".
#
# Double-checked locking pattern: the fast path (cache hit) is lock-free.
# Only a cache miss acquires cache_lock, then re-checks inside the lock to
# prevent two threads both loading the same artifact simultaneously.
from concurrency import cache_lock as _cache_lock

_model_cache = {}

def _load_cached(cache_key, loader_fn):
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    with _cache_lock():
        if cache_key not in _model_cache:
            _model_cache[cache_key] = loader_fn()
    return _model_cache[cache_key]

def load_model(ck):
    # TensorFlow Keras model — the heavy artifact, loaded once per client.
    return _load_cached(
        f"{ck}:mdl",
        lambda: tf.keras.models.load_model(_ap(ck, "per_pax_tf_model.keras"))
    )

def load_enc(ck, col):
    # LabelEncoder for a single feature column.
    return _load_cached(
        f"{ck}:enc:{col}",
        lambda: joblib.load(_ap(ck, f"{col}_encoder.pkl"))
    )

def load_map(ck, name):
    # Lookup dict artifact (item→subcat, item→cat, cat→subs).
    return _load_cached(
        f"{ck}:map:{name}",
        lambda: joblib.load(_ap(ck, f"{name}.pkl"))
    )


# ── Safe label encoding ───────────────────────────────────────────────────────

def encode_safe(le, v, default=0):
    """
    Encode v with le, falling back to 'other'/'unknown' buckets for OOV values.
    The _classes_set attribute is cached on the encoder to avoid repeated set()
    construction on every prediction call.
    """
    v = token(v)
    known = getattr(le, "_classes_set", None)
    if known is None:
        known = set(getattr(le, "classes_", []))
        setattr(le, "_classes_set", known)
    if v in known:
        return int(le.transform([v])[0])
    if "other" in known:
        return int(le.transform(["other"])[0])
    if "unknown" in known:
        return int(le.transform(["unknown"])[0])
    return default

def resolve_ci(name, known_items):
    """Case-insensitive lookup of an item name against the known item set."""
    lower_map = {k.lower(): k for k in known_items}
    return lower_map.get(name.lower(), name)

def fallback_by_sub(subcat, item_enc, item_to_subcat):
    """
    When a prediction item is not in the training vocabulary, find any other
    item that belongs to the same sub-category as a stand-in.
    Returns the item name string, or None if no match exists.
    """
    target_sub = norm(subcat)
    for item, sc in item_to_subcat.items():
        if norm(sc) == target_sub and item in item_enc.classes_:
            return item
    return None


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(ck, path, logic):
    """
    Read the client's Excel dataset and apply client-specific canonicalisation
    to category/day_type/holiday_type/meal_day columns via the logic dispatch map.
    Some clients store training data on a non-default sheet (e.g. Clario uses
    'wastage'); the optional 'sheet' key in CLIENT_DB handles that.
    """
    from client_database import CLIENT_DB
    sheet = CLIENT_DB.get(ck, {}).get("sheet", 0)
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Each client may use different spellings/casing for categories and day types.
    # The logic object provides canonical forms so the encoder sees consistent tokens.
    canonicalise = {
        "category":     logic.canonicalize_category,
        "day_type":     logic.canonicalize_day_type,
        "holiday_type": logic.canonicalize_holiday_type,
        "meal_day":     logic.canonicalize_meal_day,
    }
    for col in ["menu_items", "sub_category", "category", "day_type", "holiday_type", "meal_day", "meal_type"]:
        if col in df.columns:
            df[col] = df[col].map(canonicalise.get(col, norm))
    return df

def build_cat_to_subs(item_to_cat, item_to_subcat):
    """Build a category → [sub-categories] lookup used by the UI for hints."""
    result = defaultdict(set)
    for item, cat in item_to_cat.items():
        sub = item_to_subcat.get(item)
        if cat and sub:
            result[norm(cat)].add(norm(sub))
    return {k: sorted(v) for k, v in result.items()}

def get_nv_cats(ck, path, logic):
    """Return the list of categories that appear as non-veg in this client's data."""
    df = load_dataset(ck, path, logic)
    if "meal_type" not in df.columns:
        return []
    return sorted(df.loc[df["meal_type"] == "non veg", "category"].dropna().unique())


# ── Model training ────────────────────────────────────────────────────────────

def train_model(ck, path, logic, verbose=0):
    np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)
    df = load_dataset(ck, path, logic)

    # ── Schema validation ─────────────────────────────────────────────────────
    required = {"date", "menu_items", "sub_category", "category", "ideal_pp"}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise ValueError(
            f"[{ck}] Dataset missing required columns: {missing_required}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    # These optional columns are silently filled with safe defaults when absent.
    autofill_defaults = {
        "day_type":     "regular",
        "holiday_type": "not applicable",
        "meal_day":     "veg",
    }
    # weekday is always derived from date, so it's excluded from this check.
    missing_encoder = [
        c for c in logic.encoder_columns
        if c not in df.columns and c not in ("weekday",) and c not in autofill_defaults
    ]
    if missing_encoder:
        raise ValueError(
            f"[{ck}] Missing encoder columns required by logic: {missing_encoder}. "
            f"Either add them to the dataset or remove them from encoder_columns."
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(df["date"].isna().sum())
    if bad_dates > 0:
        raise ValueError(f"[{ck}] 'date' column has {bad_dates} unparseable values.")

    df["ideal_pp"] = pd.to_numeric(df["ideal_pp"], errors="coerce")
    bad_pp = int(df["ideal_pp"].isna().sum())
    if bad_pp > 0:
        raise ValueError(f"[{ck}] 'ideal_pp' column has {bad_pp} non-numeric values.")

    for col in ["menu_items", "sub_category", "category"]:
        df[col] = df[col].map(token).astype(str)

    # Derive weekday from date (not stored in most datasets).
    df["weekday"] = df["date"].dt.day_name().map(norm)

    for col in logic.encoder_columns:
        if col not in df.columns and col in autofill_defaults:
            df[col] = autofill_defaults[col]

    # ── Build item lookup maps ────────────────────────────────────────────────
    # These dicts map item name → sub-category / category. They are saved as
    # artifacts so the prediction path can resolve sub-cat/cat for any new item
    # entered at runtime (without re-reading the dataset).
    item_to_subcat = df.set_index("menu_items")["sub_category"].to_dict()
    item_to_cat    = df.set_index("menu_items")["category"].to_dict()
    joblib.dump(item_to_subcat, _ap(ck, "item_to_subcat.pkl"))
    joblib.dump(item_to_cat,    _ap(ck, "item_to_cat.pkl"))
    joblib.dump(build_cat_to_subs(item_to_cat, item_to_subcat), _ap(ck, "cat_to_subs.pkl"))

    # ── Fit label encoders ────────────────────────────────────────────────────
    # One sklearn LabelEncoder per feature column converts string tokens to
    # integer indices consumed by the Keras Embedding layers.
    # An "other" token is always appended so the encoder has an OOV bucket for
    # values seen at prediction time that were absent from training data.
    # encode_safe() routes anything unknown here instead of raising.
    encoders = {}
    for col in logic.encoder_columns:
        values = df[col].map(token).astype(str)
        le = LabelEncoder()
        le.fit(pd.concat([values, pd.Series(["other"])], ignore_index=True))
        df[f"{col}_idx"] = le.transform(values)
        joblib.dump(le, _ap(ck, f"{col}_encoder.pkl"))
        encoders[col] = le

    # ── Build co-menu context vectors ─────────────────────────────────────────
    # For each row, record the embedding indices of all OTHER items served on
    # the same date. This "co-menu context" lets the model learn that a dish's
    # per-pax shifts depending on what else is on the menu — e.g. rice quantity
    # drops when noodles are also served that day.
    #
    # Sequences are truncated to CTX_LEN and right-padded with pad_idx.
    # pad_idx = vocab_size (one beyond the last real item) is the OOV slot
    # inside the Keras Embedding layer (hence vocab+1 in the layer definition).
    item_to_idx  = {item: idx for idx, item in enumerate(encoders["menu_items"].classes_)}
    pad_idx      = len(item_to_idx)
    items_by_date = df.groupby("date")["menu_items"].apply(list)

    def _build_co_menu_ctx(row):
        other_indices = [
            item_to_idx[it]
            for it in items_by_date[row["date"]]
            if it in item_to_idx and it != row["menu_items"]
        ]
        truncated = other_indices[:CTX_LEN]
        return truncated + [pad_idx] * (CTX_LEN - len(truncated))

    df["co_menu_ctx"] = df[["date", "menu_items"]].apply(_build_co_menu_ctx, axis=1)

    # ── Assemble input arrays for Keras ───────────────────────────────────────
    # feature_cols = every encoder column except menu_items (handled separately).
    # X is a list of arrays matching the model's input layer order:
    #   [item_idx, co_menu_ctx, feature_col_1_idx, feature_col_2_idx, ...]
    feature_cols = [c for c in logic.encoder_columns if c != "menu_items"]
    X = (
        [np.array(df["menu_items_idx"]), np.array(df["co_menu_ctx"].tolist())]
        + [np.array(df[f"{c}_idx"]) for c in feature_cols]
    )
    y = np.array(df["ideal_pp"], dtype=float)

    # ── Date-grouped train / test split ───────────────────────────────────────
    # Split by unique service date rather than by individual row. All items on
    # the same day share the same co-menu context vector, so a row-level split
    # would let context items "see" their own test labels — direct data leakage.
    date_groups  = df["date"].dt.normalize().astype("string").fillna("nat")
    unique_dates = date_groups.unique().tolist()

    if len(unique_dates) >= 2:
        rng = np.random.RandomState(SEED)
        unique_dates = np.array(unique_dates, dtype=object)
        rng.shuffle(unique_dates)

        n_test_dates = max(1, int(round(len(unique_dates) * TEST_SIZE)))
        if n_test_dates >= len(unique_dates):
            n_test_dates = len(unique_dates) - 1  # always keep at least one train date

        test_dates = set(unique_dates[:n_test_dates])
        test_mask  = date_groups.isin(test_dates).to_numpy()
        train_mask = ~test_mask

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            # Degenerate date split (e.g. all rows on the same date) — fall back
            # to a plain row-level shuffle so training can still proceed.
            idx = np.arange(len(df))
            rng.shuffle(idx)
            cut = max(1, int(round(len(idx) * (1 - TEST_SIZE))))
            if cut >= len(idx):
                cut = len(idx) - 1
            train_idx, test_idx = idx[:cut], idx[cut:]
            X_train = [arr[train_idx] for arr in X]
            X_test  = [arr[test_idx]  for arr in X]
            y_train, y_test = y[train_idx], y[test_idx]
        else:
            X_train = [arr[train_mask] for arr in X]
            X_test  = [arr[test_mask]  for arr in X]
            y_train, y_test = y[train_mask], y[test_mask]
    else:
        # Single-date dataset — row-level shuffle only option.
        idx = np.arange(len(df))
        rng = np.random.RandomState(SEED)
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * (1 - TEST_SIZE))))
        if cut >= len(idx):
            cut = len(idx) - 1
        train_idx, test_idx = idx[:cut], idx[cut:]
        X_train = [arr[train_idx] for arr in X]
        X_test  = [arr[test_idx]  for arr in X]
        y_train, y_test = y[train_idx], y[test_idx]

    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError(
            f"[{ck}] Not enough rows ({len(df)}) to produce both train and test splits. "
            "Need at least 2 valid rows."
        )

    # ── Build Keras model (TensorFlow) ────────────────────────────────────────
    # The architecture is constructed dynamically from this client's feature schema
    # so that adding/removing encoder columns in client_logic.py automatically
    # changes the model shape — no manual layer editing needed.
    #
    # Input streams:
    #   item        → Embedding(vocab+1, 8) → Flatten
    #                 Learns a dense vector for the target dish.
    #   co_menu_ctx → Embedding(vocab+1, 8) → GlobalAveragePooling1D
    #                 Same vocabulary; averages the embeddings of all co-menu items
    #                 into a single "menu context" vector.
    #   feature cols → one Embedding per column (weekday, day_type, holiday_type, …)
    #                  Dim per column is set by EMB_DIMS above.
    #
    # All streams are concatenated → Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dense(1)
    #
    # TUNING (head size): 64 → 32 → 1 with 30% dropout. Deliberately small —
    # the concatenated embedding is only ~30 wide and the datasets are a few
    # thousand rows, so a wider head memorises the training menu instead of
    # learning it. Dropout 0.3 is the main regulariser; loss is plain MSE on
    # ideal_pp (kg/person), and Adam runs at its Keras default learning rate.
    vocab_size = len(encoders["menu_items"].classes_)

    # vocab_size + 1 everywhere below: index `vocab_size` is the pad/OOV slot.
    inp_item = tf.keras.Input(shape=(1,),       name="item")
    inp_ctx  = tf.keras.Input(shape=(CTX_LEN,), name="ctx")   # co-menu context
    inputs   = [inp_item, inp_ctx]

    # Two independent Embedding layers share the same vocabulary but learn
    # separate weight matrices — the item and its context are treated differently.
    embeddings = [
        tf.keras.layers.Flatten()(
            tf.keras.layers.Embedding(vocab_size + 1, 8)(inp_item)
        ),
        tf.keras.layers.GlobalAveragePooling1D()(
            tf.keras.layers.Embedding(vocab_size + 1, 8)(inp_ctx)
        ),
    ]

    for col in feature_cols:
        inp = tf.keras.Input(shape=(1,), name=col)
        inputs.append(inp)
        dim = EMB_DIMS.get(col, 2)
        embeddings.append(
            tf.keras.layers.Flatten()(
                tf.keras.layers.Embedding(len(encoders[col].classes_) + 1, dim)(inp)
            )
        )

    # MLP head
    x   = tf.keras.layers.Concatenate()(embeddings)
    x   = tf.keras.layers.Dense(64, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")

    # ── Train, evaluate, save ─────────────────────────────────────────────────
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH,
              validation_split=VAL_SPLIT, verbose=verbose)
    preds = model.predict(X_test, verbose=0).flatten()
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    model.save(_ap(ck, "per_pax_tf_model.keras"))
    return model, rmse


# ── Prediction ────────────────────────────────────────────────────────────────

class PredResult:
    __slots__ = ("per_pax", "total_qty", "fallback", "fallback_item", "error")

    def __init__(self, pp=None, tq=None, fb=False, fi=None, err=None):
        self.per_pax      = pp
        self.total_qty    = tq
        self.fallback     = fb    # True when a proxy item was used instead of the actual item
        self.fallback_item = fi   # which proxy item was used
        self.error        = err

    @property
    def ok(self):
        return self.per_pax is not None


def predict(ck, item, menu, mg, subcat, cat, weekday, logic,
            day_type=None, holiday_type=None, meal_day=None):
    model          = load_model(ck)
    item_enc       = load_enc(ck, "menu_items")
    item_to_subcat = load_map(ck, "item_to_subcat")
    feature_cols   = [c for c in logic.encoder_columns if c != "menu_items"]

    item_token   = (item or "").strip()
    menu_tokens  = [(i or "").strip() for i in menu]

    # Build the set of known items (cached on the encoder object to avoid
    # re-constructing the set on every call).
    known_items = getattr(item_enc, "_classes_set", None)
    if known_items is None:
        known_items = set(item_enc.classes_)
        setattr(item_enc, "_classes_set", known_items)

    pad_idx    = len(item_enc.classes_)  # OOV embedding slot (vocab_size index)
    item_token = resolve_ci(item_token, known_items)

    # ── Item lookup with sub-category fallback ────────────────────────────────
    # If the user enters an item the model has never seen, we look for another
    # item from the same sub-category as a stand-in. This handles new dishes
    # added to the menu after the model was trained.
    is_fallback   = False
    fallback_item = None
    if item_token not in known_items:
        fallback_item = fallback_by_sub(norm(subcat), item_enc, item_to_subcat)
        if fallback_item:
            is_fallback = True
            item_idx = int(item_enc.transform([fallback_item])[0])
        else:
            return PredResult(err=f"No fallback for sub-category '{subcat}'.")
    else:
        item_idx = int(item_enc.transform([item_token])[0])

    # ── Co-menu context for this prediction ───────────────────────────────────
    # Collect indices of the other known items on today's menu (excluding the
    # target item itself), then pad to CTX_LEN exactly as done during training.
    ctx_indices = []
    for menu_item in menu_tokens:
        if menu_item.lower() == item_token.lower():
            continue
        resolved = resolve_ci(menu_item, known_items)
        if resolved in known_items:
            ctx_indices.append(int(item_enc.transform([resolved])[0]))
    co_menu_ctx = ctx_indices[:CTX_LEN] + [pad_idx] * (CTX_LEN - len(ctx_indices[:CTX_LEN]))

    # ── Encode extra categorical features ─────────────────────────────────────
    # Must follow the same column order used when building X during training.
    feature_values = {
        "sub_category":  norm(subcat),
        "category":      logic.canonicalize_category(cat),
        "weekday":       norm(weekday),
        "day_type":      logic.canonicalize_day_type(day_type or "regular"),
        "holiday_type":  logic.canonicalize_holiday_type(holiday_type or "not applicable"),
        "meal_day":      logic.canonicalize_meal_day(meal_day or "veg"),
    }

    # Load all encoders before the loop — one cache lookup per column, not one per iteration.
    encoders = {col: load_enc(ck, col) for col in feature_cols}

    feed = [np.array([item_idx]), np.array([co_menu_ctx])]
    for col in feature_cols:
        feed.append(np.array([encode_safe(encoders[col], feature_values.get(col, ""))]))

    # ── Run inference ─────────────────────────────────────────────────────────
    raw = model.predict(feed, verbose=0)[0][0]
    pp  = floor005(float(raw))   # round down to 0.005 granularity
    tq  = pp * mg                # total quantity = per-pax × meal group count
    return PredResult(pp=pp, tq=tq, fb=is_fallback, fi=fallback_item)
