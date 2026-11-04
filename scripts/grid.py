"""
Grid search: train every combination of GRID for one client (on top of its
current settings) and score each on two things:
  internal_rmse  the dataset's own held-out dates — use this to CHOOSE
  case_accuracy  the client's test-case days — use this to CONFIRM
Choosing on the test days alone flatters the winner: with this many tries the
top scorer is often just a lucky random start (about ±1.2% for Clario).

Writes reports/<client>/grid-<time>.csv, one row per combination, saved as it
goes so an interrupted run keeps its results. Production settings are untouched.

Usage:
    python scripts/grid.py --client clario                       # settings grid
    python scripts/grid.py --client clario --grid architecture --set mask_padding=true --set dropout=0.1
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from datetime import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("ARTIFACT_DIR", os.path.join(_ROOT, "artifacts"))

import ml_core                                       # noqa: E402
import test_model                                    # noqa: E402
from client_logic import get_logic                   # noqa: E402

# ═══ TUNING: what each grid varies (everything else stays at the client's settings) ═══
SETTINGS_GRID = {
    "mask_padding":   [False, True],
    "extra_features": [[], ["month", "month_part", "mg_band"]],
    "dropout":        [0.1, 0.3, 0.5],
    "item_emb_dim":   [4, 8],
    "learning_rate":  [0.001, 0.003, 0.01],
    "dense_units":    [[32, 16], [64, 32]],
}
ARCHITECTURE_GRID = {
    "loss":       ["mse", "huber", "pseudo_huber"],
    "optimizer":  ["adam", "adamw"],
    "activation": ["relu", "leaky_relu", "elu"],
    "batch_norm": [False, True],
}
# softplus can only run in raw mode (category_offset predicts negative gaps).
OUTPUT_GRID = {
    "target_mode":       ["raw"],
    "output_activation": ["linear", "softplus"],
}
GRIDS = {"settings": SETTINGS_GRID, "architecture": ARCHITECTURE_GRID, "output": OUTPUT_GRID}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--client", required=True, help="Client key, e.g. clario.")
    parser.add_argument("--grid", choices=GRIDS, default="settings", help="Which grid to run.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                        help="Start from the client's settings with this change, e.g. dropout=0.1.")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    ck, L = args.client, get_logic(args.client)
    cases = test_model.read_frame(test_model.ensure_workbook(ck, L)[0], "cases")
    base = {**ml_core.get_params(ck), **{k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}}
    ml_core.validate_params(base, "--set")
    grid = GRIDS[args.grid]
    out = os.path.join(test_model.REPORT_DIR, ck, f"grid-{args.grid}-{datetime.now():%Y%m%d-%H%M%S}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    combos = list(itertools.product(*grid.values()))
    fields = list(grid) + ["internal_rmse", "case_accuracy", "within_10", "best_epoch", "epochs_run"]

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, combo in enumerate(combos, 1):
            t0 = time.time()
            setting = dict(zip(grid, combo))
            rmse, rows, hist, _ = test_model.trial(ck, L, {**base, **setting}, cases)
            ok = [r for r in rows if r.get("pred") is not None]
            val = hist["val_loss"]
            row = {**{k: ",".join(map(str, v)) if isinstance(v, list) else v for k, v in setting.items()},
                   "internal_rmse": round(rmse, 5),
                   "case_accuracy": round(sum(r["acc"] for r in ok) / len(ok), 2) if ok else None,
                   "within_10": sum(r["within"] for r in ok),
                   "best_epoch": val.index(min(val)) + 1, "epochs_run": len(val)}
            writer.writerow(row)
            f.flush()
            print(f"[{i:>3}/{len(combos)}] rmse {row['internal_rmse']:.4f}  acc {row['case_accuracy']}%  "
                  f"{setting}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
