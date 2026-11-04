"""
Per-client hyperparameter search — writes model_params.json.

For each client, tries DEFAULT_PARAMS plus N random configurations and keeps
the one with the lowest validation RMSE. Never looks at the held-out test
dates (ml_core.split_train_test): candidates are scored on a separate slice
of the TRAINING dates, so test_model.py's test-set numbers stay honest.

Each candidate trains with early stopping; the tuned `epochs` is the epoch
where validation loss bottomed out.

Usage:
    python scripts/tune.py --client clario --staged # one setting at a time, priority order
    python scripts/tune.py                          # all embedding clients
    python scripts/tune.py --client tekion --trials 24
Then retrain with the tuned params:
    python scripts/train.py --client all --force
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("ARTIFACT_DIR", os.path.join(_ROOT, "artifacts"))

import numpy as np                                   # noqa: E402
import tensorflow as tf                              # noqa: E402

import ml_core                                       # noqa: E402
from client_database import CLIENT_DB                # noqa: E402
from client_logic import get_logic                   # noqa: E402

# ═══ TUNING: search space ═════════════════════════════════════════════════════
SPACE = {
    "batch":         [16, 32, 64],
    "learning_rate": [0.0003, 0.001, 0.003, 0.01],
    "item_emb_dim":  [4, 8, 16],
    "dense_units":   [[32, 16], [64, 32], [128, 64]],
    "dropout":       [0.1, 0.2, 0.3, 0.5],
    "target_mode":   ["raw", "category_offset"],
    "mask_padding":  [False, True],
}
MAX_EPOCHS = ml_core.DEFAULT_PARAMS["epochs"]   # upper bound; early stopping ends far sooner
VAL_DATES  = 0.2   # fraction of TRAINING dates used to score candidates

# --staged: one setting at a time, in priority order, starting from the client's
# current settings. A new value is kept only if it lowers validation RMSE by more
# than MIN_GAIN (0.5%): smaller gains are within run-to-run noise.
STAGES = [
    ("early_stopping", [3, 5, 8]),
    ("dropout",        [0.1, 0.2, 0.3, 0.4, 0.5]),
    ("item_emb_dim",   [4, 8, 16]),
    ("learning_rate",  [0.0003, 0.001, 0.003, 0.01]),
    ("dense_units",    [[32, 16], [64, 32], [128, 64]]),
    ("batch",          [16, 32, 64]),
]
MIN_GAIN = 0.005


def inner_split(df, train_idx):
    """Split the training rows by date into (fit_idx, val_idx)."""
    dates = df["date"].dt.normalize().astype("string").to_numpy()
    train_dates = np.array(sorted(set(dates[train_idx])), dtype=object)
    np.random.RandomState(ml_core.SEED + 1).shuffle(train_dates)
    n_val = max(1, int(round(len(train_dates) * VAL_DATES)))
    val_dates = set(train_dates[:n_val])
    is_val = np.array([dates[i] in val_dates for i in train_idx])
    return train_idx[~is_val], train_idx[is_val]


def score(params, X, y, fit_idx, val_idx, encoders, feature_cols, base):
    """Train one candidate with its own early stopping; return (val RMSE, best epoch).

    base = per-row category mean from the fit rows; subtracted for category_offset.
    The val RMSE of the residual equals the val RMSE of ideal_pp itself.
    """
    if params["target_mode"] == "category_offset":
        y = y - base
    seed = int(params["seed"])
    np.random.seed(seed); tf.random.set_seed(seed); random.seed(seed)
    model = ml_core.build_model(encoders, feature_cols, params)
    stop = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(params["early_stopping"]),
                                             restore_best_weights=True)]
    hist = model.fit([a[fit_idx] for a in X], y[fit_idx],
                     validation_data=([a[val_idx] for a in X], y[val_idx]),
                     epochs=int(params["epochs"]),   # a maximum; early stopping ends sooner
                     batch_size=int(params["batch"]),
                     callbacks=stop, verbose=0)
    val = hist.history["val_loss"]
    best = int(np.argmin(val))
    return float(np.sqrt(val[best])), best + 1


def _setup(ck, params=None):
    """Encoded data and the fit / validation split used to score candidates."""
    info, L = CLIENT_DB[ck], get_logic(ck)
    params = params or ml_core.get_params(ck)
    df = ml_core.prepare_dataset(ck, info["dataset"], L)
    columns = ml_core.model_columns(L, params)   # extra_features are chosen, not tuned
    encoders = ml_core.fit_encoders(df, columns)
    X, y = ml_core.build_inputs(df, columns, encoders)
    feature_cols = [c for c in columns if c != "menu_items"]
    # Test dates are never used here; test-case days are already dropped by prepare_dataset.
    train_idx, _test_idx = ml_core.split_train_test(df, params["test_size"])
    fit_idx, val_idx = inner_split(df, train_idx)
    fit_means = df.iloc[fit_idx].groupby("category")["ideal_pp"].mean()
    base = df["category"].map(fit_means).fillna(df.iloc[fit_idx]["ideal_pp"].mean()).to_numpy(dtype=float)
    return X, y, fit_idx, val_idx, encoders, feature_cols, base


def tune_staged(ck):
    """Tune one setting at a time in STAGES order; keep a change only if it clearly helps."""
    data = _setup(ck)
    current = ml_core.get_params(ck)

    def run(p):
        return score(p, *data)

    cur_rmse, _ = run(current)
    start_rmse = cur_rmse
    print(f"  [{ck}] start (current settings): val RMSE {cur_rmse:.4f}")
    for knob, values in STAGES:
        best = None
        for v in values:
            if v == current[knob]:
                continue
            cand = {**current, knob: v}
            rmse, epoch = run(cand)
            print(f"  [{ck}] {knob}={v}: val RMSE {rmse:.4f} @ epoch {epoch}")
            if best is None or rmse < best[0]:
                best = (rmse, cand)
        if best and best[0] < cur_rmse * (1 - MIN_GAIN):
            print(f"  [{ck}] keep {knob}={best[1][knob]} ({cur_rmse:.4f} → {best[0]:.4f})")
            cur_rmse, current = best
        else:
            print(f"  [{ck}] keep {knob}={current[knob]} (no clear gain)")
    tuned = dict(current)
    tuned.update(_val_rmse=round(cur_rmse, 5), _start_val_rmse=round(start_rmse, 5),
                 _tuned_at=datetime.now().strftime("%Y-%m-%d %H:%M"), _mode="staged")
    print(f"[ok] {ck}: val RMSE {start_rmse:.4f} (current) → {cur_rmse:.4f} (staged)")
    return tuned


def tune_client(ck, trials):
    X, y, fit_idx, val_idx, encoders, feature_cols, base = _setup(ck)
    rng = random.Random(ml_core.SEED)
    candidates = [dict(ml_core.DEFAULT_PARAMS)]
    seen = {json.dumps(candidates[0], sort_keys=True)}
    while len(candidates) < trials + 1:
        c = {**ml_core.DEFAULT_PARAMS, **{k: rng.choice(v) for k, v in SPACE.items()}}
        c["epochs"] = MAX_EPOCHS
        key = json.dumps({k: v for k, v in c.items() if k != "epochs"}, sort_keys=True)
        if key not in seen:
            seen.add(key); candidates.append(c)

    results = []
    for i, c in enumerate(candidates):
        t0 = time.time()
        rmse, best_epoch = score(c, X, y, fit_idx, val_idx, encoders, feature_cols, base)
        results.append((rmse, best_epoch, c))
        tag = "default" if i == 0 else f"trial {i:>2}"
        print(f"  [{ck}] {tag}: val RMSE {rmse:.4f} @ epoch {best_epoch:>2}  "
              f"lr={c['learning_rate']} batch={c['batch']} emb={c['item_emb_dim']} "
              f"dense={c['dense_units']} drop={c['dropout']} target={c['target_mode']}  ({time.time() - t0:.0f}s)")

    default_rmse = results[0][0]
    rmse, best_epoch, best = min(results, key=lambda r: r[0])
    tuned = {k: best[k] for k in ml_core.DEFAULT_PARAMS}
    tuned["epochs"] = MAX_EPOCHS   # a maximum: early stopping finds the real count
    tuned["_val_rmse"] = round(rmse, 5)
    tuned["_default_val_rmse"] = round(default_rmse, 5)
    tuned["_tuned_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    tuned["_trials"] = trials
    print(f"[ok] {ck}: val RMSE {default_rmse:.4f} (default) → {rmse:.4f} (tuned)")
    return tuned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--client", default="all", help="Client key or 'all'.")
    parser.add_argument("--trials", type=int, default=16, help="Random configs per client.")
    parser.add_argument("--staged", action="store_true",
                        help="Tune one setting at a time in priority order instead of random search.")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    targets = ([k for k, v in CLIENT_DB.items() if v.get("has_embeddings")]
               if args.client == "all" else [args.client])

    params = {}
    if os.path.exists(ml_core.PARAMS_FILE):
        with open(ml_core.PARAMS_FILE, encoding="utf-8") as f:
            params = json.load(f)

    ok = True
    for ck in targets:
        try:
            params[ck] = tune_staged(ck) if args.staged else tune_client(ck, args.trials)
            # Save after every client so a crash keeps finished work.
            with open(ml_core.PARAMS_FILE, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2)
        except Exception as e:
            ok = False
            print(f"[fail] {ck}: {e}")
    print(f"\nwrote {ml_core.PARAMS_FILE}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
