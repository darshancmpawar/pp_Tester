"""
Per-client hyperparameter search — writes model_params.json.

For each client, tries DEFAULT_PARAMS plus N random configurations and keeps
the one with the lowest validation RMSE. Never looks at the held-out test
dates (ml_core.split_train_test): candidates are scored on a separate slice
of the TRAINING dates, so test_model.py's test-set numbers stay honest.

Each candidate trains with early stopping; the tuned `epochs` is the epoch
where validation loss bottomed out.

Usage:
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
}
MAX_EPOCHS = 80    # upper bound; early stopping usually ends far sooner
PATIENCE   = 8     # epochs without val improvement before stopping
VAL_DATES  = 0.2   # fraction of TRAINING dates used to score candidates


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
    """Train one candidate with early stopping; return (val RMSE, best epoch).

    base = per-row category mean from the fit rows; subtracted for category_offset.
    The val RMSE of the residual equals the val RMSE of ideal_pp itself.
    """
    if params["target_mode"] == "category_offset":
        y = y - base
    np.random.seed(ml_core.SEED); tf.random.set_seed(ml_core.SEED); random.seed(ml_core.SEED)
    model = ml_core.build_model(encoders, feature_cols, params)
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                            restore_best_weights=True)
    hist = model.fit([a[fit_idx] for a in X], y[fit_idx],
                     validation_data=([a[val_idx] for a in X], y[val_idx]),
                     epochs=int(params["epochs"]),   # defaults: 20; random trials: MAX_EPOCHS
                     batch_size=int(params["batch"]),
                     callbacks=[stop], verbose=0)
    val = hist.history["val_loss"]
    best = int(np.argmin(val))
    return float(np.sqrt(val[best])), best + 1


def tune_client(ck, trials):
    info, L = CLIENT_DB[ck], get_logic(ck)
    df = ml_core.prepare_dataset(ck, info["dataset"], L)
    encoders = ml_core.fit_encoders(df, L)
    X, y = ml_core.build_inputs(df, L, encoders)
    feature_cols = [c for c in L.encoder_columns if c != "menu_items"]
    # Test dates are never used here; test-case days are already dropped by prepare_dataset.
    train_idx, _test_idx = ml_core.split_train_test(df, ml_core.get_params(ck)["test_size"])
    fit_idx, val_idx = inner_split(df, train_idx)
    fit_means = df.iloc[fit_idx].groupby("category")["ideal_pp"].mean()
    base = df["category"].map(fit_means).fillna(df.iloc[fit_idx]["ideal_pp"].mean()).to_numpy(dtype=float)

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
    tuned["epochs"] = best_epoch
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
            params[ck] = tune_client(ck, args.trials)
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
