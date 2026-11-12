"""
ML Core — dynamic model training + prediction for any client's feature schema.

The model architecture is built dynamically from the client's encoder_columns list.
All artifacts are prefixed with the client key (e.g. tekion_per_pax_tf_model.keras).
"""
import os, random
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
# There is one architecture, shaped per client by its encoder_columns. The
# constants in this block are shared by every client (changing them changes the
# data pipeline or the test split). The per-client knobs — epochs, batch size,
# learning rate, head size, dropout, item embedding width — live in
# model_params.json, edited by hand from what `python scripts/search.py` finds.
# A client missing from that file trains with DEFAULT_PARAMS. After editing it,
# run `python scripts/train.py`: it retrains exactly the clients whose settings,
# dataset, standard portions or test-case days changed since their last training
# (stale_reasons).

SEED       = 42   # fixed so two runs on the same dataset give the same model
CTX_LEN    = 10   # max co-menu items kept as context per row. A full menu is
                  # 8–10 dishes, so 10 covers a day without padding waste.
VAL_SPLIT  = 0.1  # fraction of TRAINING rows Keras holds out to watch val_loss
TEST_SIZE  = 0.2  # fraction of DATES held out as the final (reported) test set

# The original shared settings, kept as the fallback for untuned clients.
DEFAULT_PARAMS = {
    "epochs":        100,       # MAXIMUM: early stopping (always on) picks the real count
    "batch":         32,
    "learning_rate": 0.001,     # Keras Adam default
    "optimizer":     "adam",    # adam | adamw | rmsprop | sgd
    "loss":          "mse",     # mse | mae | huber | pseudo_huber
    "activation":    "relu",    # hidden-layer activation: relu | leaky_relu | elu | gelu | tanh
    "item_emb_dim":  8,         # width of the dish + co-menu embeddings
    "dense_units":   [64, 32],
    "dropout":       0.3,
    # raw: the model predicts ideal_pp directly.
    # category_offset: it predicts ideal_pp minus the category's training-row
    # mean, which is added back at prediction time. Categories differ by up to
    # 10× in portion (bread pieces vs rasam kg), so the network only has to
    # learn the small per-dish/per-day adjustment, not the category's scale.
    "target_mode":   "raw",
    "mask_padding":  False,     # ignore pad slots when averaging the co-menu (build_model)
    # Random starting weights only; the train/test split always uses SEED, so runs stay comparable.
    "seed":          SEED,
    # Switch point of huber / pseudo_huber, kg. Errors below it are treated like MSE
    # (chases the average), above it like MAE (chases the typical dish).
    "loss_delta":    0.05,
    # Early stopping is always on: training stops after this many epochs without a
    # better validation loss and keeps the weights from the best epoch. Those extra
    # epochs also show on the learning curve, proving the kept epoch is the best.
    "early_stopping": 5,
    # TRUE: after training, each category's typical miss on the validation rows (the newest
    # training days) is added to its predictions, so the model follows the current standard
    # portion instead of a blend of old and new ones. See category_calibration.
    "calibrate":     False,
    # TRUE: before training, every row is scaled to TODAY's standard portion for its
    # category (data/<client>_standard_portions.csv): × today's standard ÷ the standard
    # in force that month. Recomputed at every retrain. See rebase_to_standards.
    "rebase":        False,
    # How many models are trained (same data, different random starts) and averaged.
    # Averaging cancels each model's random quirks: steadier, and it cannot overfit more.
    "ensemble":      1,
    "val_split":     VAL_SPLIT,
    "test_size":     TEST_SIZE,
}
PARAMS_FILE = os.path.join(os.path.dirname(__file__), "model_params.json")

# Each client's test-case workbook (test_model.py). Every date on its "cases"
# sheet is dropped from training, tuning and evaluation data, so the model
# never sees the days it is tested on.
TEST_CASES_DIR = os.path.join(os.path.dirname(__file__), "test_cases")

def test_workbook_path(ck):
    return os.path.join(TEST_CASES_DIR, f"{ck}_test.xlsx")

def holdout_dates(ck):
    """Normalised dates on the client's test-case sheet (empty set if none)."""
    path = test_workbook_path(ck)
    if not os.path.exists(path):
        return set()
    dates = pd.to_datetime(pd.read_excel(path, sheet_name="cases")["date"], errors="coerce")
    return set(dates.dropna().dt.normalize())

PARAM_CHOICES = {
    "optimizer":   ("adam", "adamw", "rmsprop", "sgd"),
    "loss":        ("mse", "mae", "huber", "pseudo_huber"),
    "activation":  ("relu", "leaky_relu", "elu", "gelu", "tanh"),
    "target_mode": ("raw", "category_offset"),
}

def validate_params(params, where):
    """Raise ValueError naming the bad setting instead of training with a wrong value."""
    unknown = sorted(k for k in params if not k.startswith("_") and k not in DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"{where}: unknown setting(s) {unknown}. "
                         f"Valid names: {sorted(DEFAULT_PARAMS)} (prefix notes with '_').")
    for k, choices in PARAM_CHOICES.items():
        if k in params and params[k] not in choices:
            raise ValueError(f"{where}: {k}={params[k]!r} — must be one of {list(choices)}.")
    if "early_stopping" in params and not (isinstance(params["early_stopping"], int)
                                           and not isinstance(params["early_stopping"], bool)
                                           and params["early_stopping"] >= 1):
        raise ValueError(f"{where}: early_stopping={params['early_stopping']!r} — must be a whole "
                         "number of epochs, 1 or more (early stopping is always on).")
    if "loss_delta" in params and not params["loss_delta"] > 0:
        raise ValueError(f"{where}: loss_delta={params['loss_delta']!r} — must be above 0 (kg).")
    if "seed" in params and not (isinstance(params["seed"], int) and params["seed"] >= 0):
        raise ValueError(f"{where}: seed={params['seed']!r} — must be a whole number, 0 or more.")
    for k in ("mask_padding", "calibrate", "rebase"):
        if k in params and not isinstance(params[k], bool):
            raise ValueError(f"{where}: {k}={params[k]!r} — must be true or false.")
    for k in ("epochs", "batch", "item_emb_dim", "ensemble"):
        if k in params and not (isinstance(params[k], int) and params[k] > 0):
            raise ValueError(f"{where}: {k}={params[k]!r} — must be a whole number above 0.")
    if "dense_units" in params and not (isinstance(params["dense_units"], list) and len(params["dense_units"]) == 2
                                        and all(isinstance(u, int) and u > 0 for u in params["dense_units"])):
        raise ValueError(f"{where}: dense_units={params['dense_units']!r} — must be two whole numbers, e.g. [64, 32].")
    for k in ("dropout", "val_split", "test_size"):
        if k in params and not (0 <= params[k] < 1):
            raise ValueError(f"{where}: {k}={params[k]!r} — must be between 0 and 1.")
    if "learning_rate" in params and not params["learning_rate"] > 0:
        raise ValueError(f"{where}: learning_rate={params['learning_rate']!r} — must be above 0.")

def get_params(ck):
    """DEFAULT_PARAMS overlaid with this client's entry in model_params.json (validated)."""
    params = dict(DEFAULT_PARAMS)
    if os.path.exists(PARAMS_FILE):
        import json
        with open(PARAMS_FILE, encoding="utf-8") as f:
            entry = json.load(f).get(ck, {})
        validate_params(entry, f"model_params.json [{ck}]")
        params.update({k: v for k, v in entry.items() if k in DEFAULT_PARAMS})
    return params

def file_sha(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]

def standards_path(ck):
    """The kitchen's standard portions: category, weekday (blank = every day), grams, note."""
    return os.path.join(os.path.dirname(__file__), "data", f"{ck}_standard_portions.csv")

def data_sha(ck, dataset_path, params):
    """Fingerprint of what the model learns from: the dataset, plus the standards when rebase is on,
    so editing either marks the model out of date."""
    sha = file_sha(dataset_path)
    return f"{sha}+{file_sha(standards_path(ck))}" if params.get("rebase") else sha

NONVEG_DAY_ORDER = ("non-veg biryani", "non-veg gravy", "non-veg dry")   # the kitchen plans around the first

def day_kind(categories):
    """What kind of menu day it is: the non-veg category on the menu (exactly the one picked in the
    app; on a day with two, the first in NONVEG_DAY_ORDER), else "veg". The kitchen plans portions
    by it, e.g. White Rice 70 g on biryani days against 100 g otherwise."""
    cats = {token(c) for c in categories}
    return next((k for k in NONVEG_DAY_ORDER if k in cats),   # other spellings, e.g. Tekion's "non veg curry":
                next((c for c in sorted(cats) if c.replace(" ", "-").startswith("non-veg")), "veg"))

def rebase_to_standards(df, standards):
    """
    Scale each row's ideal_pp to today's standard portion: × today's standard ÷ the
    standard in force that month, taken as the month's median cooked portion (per_pax)
    for the same category and kind of day. The table's `day` column picks the row: a kind of
    day (day_kind, e.g. "non-veg biryani"), a weekday name, or blank (every other day). A 2025
    Dal day at the old 50 g standard therefore counts as a 70 g day, and a biryani day is scaled
    by the biryani-day standard. Categories without a standard keep their rows unchanged.
    """
    if "per_pax" not in df.columns:
        raise ValueError("rebase needs the per_pax (cooked per person) column in the dataset.")
    std = standards.assign(category=standards["category"].map(norm),
                           day=standards["day"].fillna("").astype(str).map(norm))
    grams = {(c, d): g / 1000.0 for c, d, g in zip(std["category"], std["day"], std["grams"])}
    df = df.copy()
    kind = df["date"].map(df.groupby("date")["category"].apply(day_kind))
    # the most specific row that applies: the kind of day, then the weekday, then every day
    wd = pd.Series([k if (c, k) in grams else w if (c, w) in grams else ""
                    for c, w, k in zip(df["category"], df["weekday"], kind)], index=df.index)
    today = pd.Series([grams.get((c, w)) for c, w in zip(df["category"], wd)], index=df.index, dtype=float)
    cooked = pd.to_numeric(df["per_pax"], errors="coerce")
    then = cooked.groupby([df["category"], wd, df["date"].dt.to_period("M")]).transform("median")
    factor = (today / then).where(lambda f: np.isfinite(f) & (f > 0), 1.0)
    df["ideal_pp"] = df["ideal_pp"] * factor
    return df

def stale_reasons(ck, dataset_path):
    """
    Why the trained model no longer matches its inputs: a list such as
    ["epochs 5→8", "dataset changed"], or [] when it is up to date. Compares the
    current settings, dataset file and test-case days with the last row of the
    client's training log.
    """
    import csv, json
    log_path = _ap(ck, "training_log.csv")
    if not os.path.exists(log_path):
        return ["no training log yet"]
    with open(log_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return ["no training log yet"]
    last, reasons = rows[-1], []
    # A setting added after that run is logged as absent; it trained with the default.
    trained = {**DEFAULT_PARAMS, **json.loads(last["params"])}
    params = get_params(ck)
    for k, v in params.items():
        if json.dumps(trained.get(k)) != json.dumps(v):
            reasons.append(f"{k} {trained.get(k)}→{v}")
    if last["dataset_sha"] != data_sha(ck, dataset_path, params):
        reasons.append("dataset or standard portions changed")
    if last.get("holdout", "") != ",".join(sorted(str(d.date()) for d in holdout_dates(ck))):
        reasons.append("test-case days changed")
    return reasons

# TUNING: embedding width per feature, in floats. Roughly "how much the model
# is allowed to memorise about this column". Wider = more capacity and more
# overfitting risk; the rule of thumb here is width ∝ cardinality:
#   menu_items    hundreds of dishes   → item_emb_dim setting (default 8), not this dict
#   sub_category  tens                 → 4
#   category      ~10                  → 4
#   weekday       5–7                  → 2
#   day_type      4, holiday_type 4, meal_day 2 → 2
# A column missing from this dict falls back to 2 (see the model builder).
EMB_DIMS = {
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

def round005(v):
    """
    Round to the nearest 0.005 kg — the per-pax output granularity. 5 g is the
    finest difference a serving spoon can hold; see the matching note in planner.py.

    Nearest, not down: on the Clario test days rounding down put fewer dishes
    within ±10% and made under-predicted dishes (shortages) worse.
    """
    return round(v * 200) / 200.0   # ×200 == ÷0.005, same as planner.round_to_005

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
        + ["item_to_subcat.pkl", "item_to_cat.pkl", "cat_to_subs.pkl", "features.pkl"]
    )
    return all(os.path.exists(_ap(ck, f)) for f in expected)


# ── In-process model cache ────────────────────────────────────────────────────
# Streamlit Cloud runs ONE Python process for all user sessions (threads).
# Loading a Keras model or encoder from disk on every prediction would be
# too slow. _model_cache holds already-loaded objects keyed by "{ck}:{type}".
# Each entry remembers the file's modification time, so a retrain while the
# app is running is picked up on the next prediction — no restart needed.
#
# Double-checked locking pattern: the fast path (cache hit) is lock-free.
# Only a cache miss acquires cache_lock, then re-checks inside the lock to
# prevent two threads both loading the same artifact simultaneously.
from concurrency import cache_lock as _cache_lock

_model_cache = {}

def _load_cached(cache_key, path, loader_fn):
    mtime = os.path.getmtime(path) if os.path.exists(path) else None
    hit = _model_cache.get(cache_key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    with _cache_lock():
        hit = _model_cache.get(cache_key)
        if hit is None or hit[0] != mtime:
            _model_cache[cache_key] = (mtime, loader_fn())
    return _model_cache[cache_key][1]

def model_file(i):
    """Model i of an ensemble: per_pax_tf_model.keras, then per_pax_tf_model_2.keras, ..."""
    return "per_pax_tf_model.keras" if i == 1 else f"per_pax_tf_model_{i}.keras"

def load_models(ck):
    # TensorFlow Keras models — the heavy artifacts, loaded once per client. Predictions
    # average all of them (one for a single model, `ensemble` of them otherwise).
    # compile=False: prediction never needs the training loss, so any loss (e.g. a
    # pseudo_huber with a custom delta) loads without being registered with Keras.
    models, i = [], 1
    while i == 1 or os.path.exists(_ap(ck, model_file(i))):
        path = _ap(ck, model_file(i))
        models.append(_load_cached(f"{ck}:mdl{i}", path,
                                   lambda p=path: tf.keras.models.load_model(p, compile=False)))
        i += 1
    return models

def load_enc(ck, col):
    # LabelEncoder for a single feature column.
    path = _ap(ck, f"{col}_encoder.pkl")
    return _load_cached(f"{ck}:enc:{col}", path, lambda: joblib.load(path))

def load_map(ck, name):
    # Lookup dict artifact (item→subcat, item→cat, cat→subs).
    path = _ap(ck, f"{name}.pkl")
    return _load_cached(f"{ck}:map:{name}", path, lambda: joblib.load(path))

def load_offsets(ck):
    # category → mean ideal_pp, present only for a category_offset model.
    path = _ap(ck, "cat_offsets.pkl")
    return _load_cached(f"{ck}:offsets", path,
                        lambda: joblib.load(path) if os.path.exists(path) else None)

def category_offset(offsets, category):
    """Training-row mean for this category, or the overall mean if unseen."""
    return offsets.get(token(category), offsets["__all__"])

def load_calibration(ck):
    # category → kg added to every prediction, present only for a model trained with calibrate.
    path = _ap(ck, "calibration.pkl")
    return _load_cached(f"{ck}:calibration", path,
                        lambda: joblib.load(path) if os.path.exists(path) else None)

CALIBRATION_SHRINK = 10   # rows: a category seen n times moves n/(n+10) of its median miss

def category_calibration(residuals, categories, shrink=CALIBRATION_SHRINK):
    """Median miss (eaten − predicted, kg) per category, shrunk toward 0 by n/(n+shrink)
    so a category with only a few validation rows barely moves."""
    g = pd.Series(np.asarray(residuals, dtype=float)).groupby(np.asarray(categories))
    med, n = g.median(), g.size()
    return {k: float(med[k] * n[k] / (n[k] + shrink)) for k in med.index}


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
        "menu_items":   logic.canonicalize_dish,
        "category":     logic.canonicalize_category,
        "day_type":     logic.canonicalize_day_type,
        "holiday_type": logic.canonicalize_holiday_type,
        "meal_day":     logic.canonicalize_meal_day,
    }
    for col in ["menu_items", "sub_category", "category", "day_type", "holiday_type", "meal_day", "meal_type"]:
        if col in df.columns:
            df[col] = df[col].map(canonicalise.get(col, norm))
    # Dishes logged under the wrong category on some days → the kitchen's category.
    for dish, (cat, sub) in getattr(logic, "dish_category_map", {}).items():
        rows = df["menu_items"] == dish
        df.loc[rows, "category"] = cat
        if "sub_category" in df.columns:
            df.loc[rows, "sub_category"] = sub
    if getattr(logic, "meal_day_from_menu", False):   # veg or the day's non-veg category, from its own menu
        df["meal_day"] = df["date"].map(df.groupby("date")["category"].apply(day_kind))
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

def prepare_dataset(ck, path, logic):
    """
    Load the client's dataset and apply every validation / cleaning step the
    model needs (required columns, dates, numeric ideal_pp, weekday, autofilled
    optional columns). Shared by train_model and test_model.py so evaluation
    sees exactly the rows training saw.
    """
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

    # Never train or evaluate on the client's test-case days.
    holdout = holdout_dates(ck)
    if holdout:
        df = df[~df["date"].dt.normalize().isin(holdout)].reset_index(drop=True)
    return df


def build_inputs(df, columns, encoders):
    """
    Encode every encoder column with the given fitted encoders and build the
    Keras input list [item_idx, co_menu_ctx, feature_col_1_idx, ...] plus y.
    Adds the *_idx and co_menu_ctx columns to df in place.
    """
    for col in columns:
        df[f"{col}_idx"] = encoders[col].transform(df[col].map(token).astype(str))

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
    feature_cols = [c for c in columns if c != "menu_items"]
    X = (
        [np.array(df["menu_items_idx"]), np.array(df["co_menu_ctx"].tolist())]
        + [np.array(df[f"{c}_idx"]) for c in feature_cols]
    )
    y = np.array(df["ideal_pp"], dtype=float)
    return X, y


def masked_mean(emb, ids, pad):
    """Average of the embeddings whose id is not `pad`; all zeros when there is none. Keras' masked
    GlobalAveragePooling1D gives 0/0 = NaN there (a one-dish menu, or every other dish new), which
    ReLU then turns into a prediction that ignores the dish, the weekday and the meal day."""
    ops = tf.keras.ops
    keep = ops.expand_dims(ops.cast(ops.not_equal(ids, pad), "float32"), -1)
    return ops.sum(emb * keep, axis=1) / ops.maximum(ops.sum(keep, axis=1), 1.0)


def split_train_test(df, test_size=TEST_SIZE):
    """
    Return (train_idx, test_idx) as integer row positions into df.

    Split by unique service date rather than by individual row. All items on
    the same day share the same co-menu context vector, so a row-level split
    would let context items "see" their own test labels — direct data leakage.
    Deterministic for a given dataset (fixed SEED).
    """
    date_groups  = df["date"].dt.normalize().astype("string").fillna("nat")
    unique_dates = date_groups.unique().tolist()
    rng = np.random.RandomState(SEED)

    if len(unique_dates) >= 2:
        unique_dates = np.array(unique_dates, dtype=object)
        rng.shuffle(unique_dates)

        n_test_dates = max(1, int(round(len(unique_dates) * test_size)))
        if n_test_dates >= len(unique_dates):
            n_test_dates = len(unique_dates) - 1  # always keep at least one train date

        test_dates = set(unique_dates[:n_test_dates])
        test_mask  = date_groups.isin(test_dates).to_numpy()
        train_mask = ~test_mask

        if train_mask.sum() > 0 and test_mask.sum() > 0:
            return np.flatnonzero(train_mask), np.flatnonzero(test_mask)
        # Degenerate date split (e.g. all rows on the same date) — fall back
        # to a plain row-level shuffle so training can still proceed.

    # Single-date dataset (or degenerate split) — row-level shuffle only option.
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = max(1, int(round(len(idx) * (1 - test_size))))
    if cut >= len(idx):
        cut = len(idx) - 1
    return idx[:cut], idx[cut:]


def fit_encoders(df, columns):
    """
    One sklearn LabelEncoder per feature column converts string tokens to
    integer indices consumed by the Keras Embedding layers.
    An "other" token is always appended so the encoder has an OOV bucket for
    values seen at prediction time that were absent from training data.
    encode_safe() routes anything unknown here instead of raising.
    """
    encoders = {}
    for col in columns:
        values = df[col].map(token).astype(str)
        le = LabelEncoder()
        le.fit(pd.concat([values, pd.Series(["other"])], ignore_index=True))
        encoders[col] = le
    return encoders


def pseudo_huber(delta):
    """Smooth Huber loss: ≈ MSE for errors under delta, ≈ MAE above, differentiable everywhere."""
    def loss(y_true, y_pred):
        r = (y_pred - y_true) / delta
        return tf.keras.ops.mean(delta ** 2 * (tf.keras.ops.sqrt(1 + r * r) - 1), axis=-1)
    return loss


def build_model(encoders, feature_cols, params):
    """
    Build and compile the Keras model for one client's feature schema + params.

    The architecture is constructed dynamically from this client's feature schema
    so that adding/removing encoder columns in client_logic.py automatically
    changes the model shape — no manual layer editing needed.

    Input streams:
      item        → Embedding(vocab+1, item_emb_dim) → Flatten
                    Learns a dense vector for the target dish.
      co_menu_ctx → Embedding(vocab+1, item_emb_dim) → average
                    Same vocabulary; averages the embeddings of all co-menu items
                    into a single "menu context" vector.
      feature cols → one Embedding per column (weekday, day_type, holiday_type, …)
                     Dim per column is set by EMB_DIMS above.

    All streams are concatenated → Dense(dense_units[0], activation) → Dropout(dropout)
    → Dense(dense_units[1], activation) → Dense(1).
    Head size, dropout, item_emb_dim and learning rate come from params
    (per client — see model_params.json / scripts/search.py), as do
    activation, optimizer and loss.
    """
    vocab_size = len(encoders["menu_items"].classes_)
    item_dim   = int(params["item_emb_dim"])

    # vocab_size + 1 everywhere below: index `vocab_size` is the pad/OOV slot.
    inp_item = tf.keras.Input(shape=(1,),       name="item")
    inp_ctx  = tf.keras.Input(shape=(CTX_LEN,), name="ctx")   # co-menu context
    inputs   = [inp_item, inp_ctx]

    # Two independent Embedding layers share the same vocabulary but learn
    # separate weight matrices — the item and its context are treated differently.
    # Layer creation order fixes the random starting weights: keep item before ctx,
    # or a retrain with unchanged settings no longer reproduces the saved model.
    item_emb = tf.keras.layers.Embedding(vocab_size + 1, item_dim)(inp_item)
    ctx_emb  = tf.keras.layers.Embedding(vocab_size + 1, item_dim)(inp_ctx)
    # mask_padding: average only the real co-menu dishes. Without it, a day with
    # 4 other dishes averages 4 real embeddings with 6 copies of the pad slot.
    embeddings = [
        tf.keras.layers.Flatten()(item_emb),
        masked_mean(ctx_emb, inp_ctx, vocab_size) if params["mask_padding"]
        else tf.keras.layers.GlobalAveragePooling1D()(ctx_emb),
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
    units = params["dense_units"]
    x   = tf.keras.layers.Concatenate()(embeddings)
    x   = tf.keras.layers.Dense(int(units[0]), activation=params["activation"])(x)
    x   = tf.keras.layers.Dropout(float(params["dropout"]))(x)
    x   = tf.keras.layers.Dense(int(units[1]), activation=params["activation"])(x)
    out = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    optimizers = {
        "adam":    tf.keras.optimizers.Adam,
        "adamw":   tf.keras.optimizers.AdamW,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "sgd":     tf.keras.optimizers.SGD,
    }
    losses = {
        "mse":   tf.keras.losses.MeanSquaredError(),
        "mae":   tf.keras.losses.MeanAbsoluteError(),
        "huber": tf.keras.losses.Huber(delta=float(params["loss_delta"])),
        "pseudo_huber": pseudo_huber(float(params["loss_delta"])),
    }
    model.compile(
        optimizer=optimizers[params["optimizer"]](learning_rate=float(params["learning_rate"])),
        loss=losses[params["loss"]],
        # reported only (training follows the loss): the real error per epoch for the learning
        # curve, since the square root of a Huber-type loss is not an RMSE
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def train_model(ck, path, logic, verbose=0, params=None, note=""):
    params = params or get_params(ck)
    seed = int(params["seed"])
    np.random.seed(seed); tf.random.set_seed(seed); random.seed(seed)
    df = prepare_dataset(ck, path, logic)
    if params["rebase"]:
        df = rebase_to_standards(df, pd.read_csv(standards_path(ck)))

    # ── Build item lookup maps ────────────────────────────────────────────────
    # These dicts map item name → sub-category / category. They are saved as
    # artifacts so the prediction path can resolve sub-cat/cat for any new item
    # entered at runtime (without re-reading the dataset).
    item_to_subcat = df.set_index("menu_items")["sub_category"].to_dict()
    item_to_cat    = df.set_index("menu_items")["category"].to_dict()
    joblib.dump(item_to_subcat, _ap(ck, "item_to_subcat.pkl"))
    joblib.dump(item_to_cat,    _ap(ck, "item_to_cat.pkl"))
    joblib.dump(build_cat_to_subs(item_to_cat, item_to_subcat), _ap(ck, "cat_to_subs.pkl"))

    columns = list(logic.encoder_columns)
    encoders = fit_encoders(df, columns)
    for col, le in encoders.items():
        joblib.dump(le, _ap(ck, f"{col}_encoder.pkl"))
    joblib.dump(columns, _ap(ck, "features.pkl"))   # predict() feeds exactly these

    X, y = build_inputs(df, columns, encoders)
    feature_cols = [c for c in columns if c != "menu_items"]

    # ── Date-grouped train / test split (see split_train_test) ────────────────
    train_idx, test_idx = split_train_test(df, float(params["test_size"]))
    X_train = [arr[train_idx] for arr in X]
    X_test  = [arr[test_idx]  for arr in X]
    y_train, y_test = y[train_idx], y[test_idx]

    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError(
            f"[{ck}] Not enough rows ({len(df)}) to produce both train and test splits. "
            "Need at least 2 valid rows."
        )

    # ── Target: raw ideal_pp, or ideal_pp minus its category's mean ───────────
    # Means come from TRAIN rows only so the test set stays unseen. The
    # offsets artifact doubles as the mode flag for predict(): present means
    # category_offset, absent means raw (a stale one is removed).
    offsets_path = _ap(ck, "cat_offsets.pkl")
    base = np.zeros(len(df))
    if params["target_mode"] == "category_offset":
        train_df = df.iloc[train_idx]
        offsets = {k: float(v) for k, v in train_df.groupby("category")["ideal_pp"].mean().items()}
        offsets["__all__"] = float(train_df["ideal_pp"].mean())
        joblib.dump(offsets, offsets_path)
        base = df["category"].map(lambda c: category_offset(offsets, c)).to_numpy(dtype=float)
    elif os.path.exists(offsets_path):
        os.remove(offsets_path)
    base_train, base_test = base[train_idx], base[test_idx]

    # ── Train, evaluate, save ─────────────────────────────────────────────────
    # `ensemble` models on the same data, model i starting from seed + i - 1 (model 1 from
    # the seed set above, so ensemble 1 is exactly the single model). Predictions average them.
    patience = int(params["early_stopping"])
    models, fits = [], []
    for i in range(1, int(params["ensemble"]) + 1):
        if i > 1:
            np.random.seed(seed + i - 1); tf.random.set_seed(seed + i - 1); random.seed(seed + i - 1)
        model = build_model(encoders, feature_cols, params)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                                      restore_best_weights=True)]
        fit = model.fit(X_train, y_train - base_train, epochs=int(params["epochs"]), callbacks=callbacks,
                        batch_size=int(params["batch"]),
                        validation_split=float(params["val_split"]), verbose=verbose)
        fits.append(fit.history)
        models.append(model)
    averaged = lambda Xs: np.mean([m.predict(Xs, verbose=0).flatten() for m in models], axis=0)

    # ── Calibration on the validation rows (Keras holds out the LAST val_split of the
    # training rows, the newest days) — a stale one is removed, like the offsets ─────────
    calib_path, calibration = _ap(ck, "calibration.pkl"), {}
    if params["calibrate"]:
        val_idx = train_idx[int(len(train_idx) * (1 - float(params["val_split"]))):]
        val_pred = averaged([a[val_idx] for a in X]) + base[val_idx]
        calibration = category_calibration(y[val_idx] - val_pred, df["category"].iloc[val_idx])
        joblib.dump(calibration, calib_path)
    elif os.path.exists(calib_path):
        os.remove(calib_path)
    preds = (averaged(X_test) + base_test
             + df["category"].iloc[test_idx].map(calibration).fillna(0.0).to_numpy(dtype=float))
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    for i, m in enumerate(models, 1):
        m.save(_ap(ck, model_file(i)))
    i = len(models) + 1
    while os.path.exists(_ap(ck, model_file(i))):   # a smaller ensemble leaves no stale extra models
        os.remove(_ap(ck, model_file(i)))
        i += 1
    model = models[0]
    # Per-epoch loss / val_loss, read by test_model.py for learning curves.
    # holdout_dates records which test-case days were kept out of training.
    history = {k: [float(v) for v in vs] for k, vs in fits[0].items()}   # the learning curve shows model 1
    # What only training knows about each averaged model; their errors are measured live
    # from the model files (test_model.evaluate_current), so they cannot go stale.
    history["models"] = [{"seed": seed + i, "epochs_run": len(h["loss"]),
                          "best_epoch": int(np.argmin(h["val_loss"])) + 1} for i, h in enumerate(fits)]
    history["holdout_dates"] = sorted(str(d.date()) for d in holdout_dates(ck))
    # Spread of what the model actually learned (raw ideal_pp, or the gap from
    # the category mean): test_model.fit_diagnosis compares val loss with it.
    history["target_std"] = float(np.std(y_train - base_train))
    # fit_diagnosis needs these to tell "stopped early at the best epoch" from "ran out of epochs".
    history["early_stopping"] = patience
    history["max_epochs"] = int(params["epochs"])
    joblib.dump(history, _ap(ck, "fit_history.pkl"))
    log_training(ck, path, df, train_idx, test_idx, params, rmse, note)
    return model, rmse


def log_training(ck, path, df, train_idx, test_idx, params, rmse, note=""):
    """
    Append one row per training run to <ck>_training_log.csv next to the model,
    so every retrain records which data, settings and test error produced it,
    what changed since the previous run (and what did not), plus an optional
    free-text note. Settings tests train into a temporary ARTIFACT_DIR, so only
    production retrains persist here.
    """
    import csv, json
    from datetime import datetime
    sha = data_sha(ck, path, params)
    holdout = sorted(str(d.date()) for d in holdout_dates(ck))
    log_path = _ap(ck, "training_log.csv")

    # ── What changed vs the previous logged run ───────────────────────────────
    prev = None
    if os.path.exists(log_path):
        with open(log_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        prev = rows[-1] if rows else None
    if prev is None:
        changes = "first logged run"
    else:
        changed, same = [], []
        old_params = {**DEFAULT_PARAMS, **json.loads(prev["params"])}
        (changed if prev["dataset_sha"] != sha else same).append(
            f"dataset ({prev['rows']}→{len(df)} rows)" if prev["dataset_sha"] != sha else "dataset")
        (changed if prev.get("holdout", "") != ",".join(holdout) else same).append("test-case days")
        for k in sorted(params):
            if json.dumps(old_params.get(k)) != json.dumps(params[k]):
                changed.append(f"{k} {old_params.get(k)}→{params[k]}")
            else:
                same.append(k)
        changes = ("changed: " + ("; ".join(changed) if changed else "nothing")
                   + " | unchanged: " + ", ".join(same))

    row = {
        "trained_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":      os.path.basename(path),
        "dataset_sha":  sha,
        "rows":         len(df),
        "days":         df["date"].dt.normalize().nunique(),
        "first_date":   str(df["date"].min().date()),
        "last_date":    str(df["date"].max().date()),
        "holdout_days": len(holdout),
        "train_rows":   len(train_idx),
        "test_rows":    len(test_idx),
        "test_rmse":    round(rmse, 5),
        "rmse_change":  "" if prev is None else f"{rmse - float(prev['test_rmse']):+.5f}",
        "changes":      changes,
        "note":         note,
        "params":       json.dumps(params, sort_keys=True),
        "holdout":      ",".join(holdout),
    }
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


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
            day_type=None, holiday_type=None, meal_day=None, service_date=None):
    models         = load_models(ck)
    item_enc       = load_enc(ck, "menu_items")
    item_to_subcat = load_map(ck, "item_to_subcat")
    feature_cols   = [c for c in load_map(ck, "features") if c != "menu_items"]

    item_token   = logic.canonicalize_dish(item or "")           # same spelling fixes as training
    menu_tokens  = [logic.canonicalize_dish(i or "") for i in menu]
    cat, subcat  = getattr(logic, "dish_category_map", {}).get(item_token, (cat, subcat))   # same category fixes

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
    # ensemble: their average. A direct call, not model.predict: same numbers, ~6x faster for one row.
    raw = float(np.mean([np.asarray(m(feed, training=False))[0, 0] for m in models]))
    offsets = load_offsets(ck)
    if offsets is not None:      # category_offset model: add the category mean back
        raw += category_offset(offsets, feature_values["category"])
    calibration = load_calibration(ck)
    if calibration:              # calibrate: the category's typical miss on the newest days
        raw += calibration.get(token(feature_values["category"]), 0.0)
    pp  = round005(raw)          # 5 g grid; internal test-set metrics still use the raw output
    tq  = pp * mg                # total quantity = per-pax × meal group count
    return PredResult(pp=pp, tq=tq, fb=is_fallback, fi=fallback_item)
