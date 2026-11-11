"""
Reliable model search for one client: every setting on its own against production,
then every combination of the settings that won.

Scored on the client's test sets (the `test` column of test_cases/<client>_test.xlsx,
dates never trained on). The LAST test set stays sealed: it is never used to choose,
only opened in the final report, so the winner is not picked on the days it is judged on.

Kitchen cost = SUM over dishes of  MG^p × (SHORTAGE_WEIGHT × kg/person short + kg/person over)
  SHORTAGE_WEIGHT 1.5 (set 2026-11-11: a little waste is acceptable, people going short is not).
  p  MG scaling: 0 = per person, 0.5 = √MG, 1 = kg of food. A change must lower the cost
     under all three, so no winner depends on the scaling. 1.0× and 2.0× are shown for comparison.

A change is kept only if, averaged over SEEDS:
  within 5 g    at least MIN_GAIN more dishes (the kitchen's target: one serving-spoon step)
  within ±10%   no fewer dishes (guards the big dishes, where 5 g is under 2%)
  kitchen cost  at least COST_GAIN lower under every MG scaling
and its internal test RMSE is no more than RMSE_TOL worse. The report also counts the days
on which more than DAY_TARGET of the dishes are within 5 g.

Every run also predicts the test days with each day's co-menu swapped for another
day's: if accuracy does not drop, the model is not reading the other dishes.

Each run's per-dish predictions are cached in reports/<client>/search_runs.jsonl (keyed
by settings, seed, dataset and test cases), so a re-run only trains what changed and the
cost can be re-weighted without retraining. model_params.json is not touched.

Usage:
    python scripts/search.py --client clario
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("ARTIFACT_DIR", os.path.join(_ROOT, "artifacts"))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

import ml_core                                       # noqa: E402
import test_model                                    # noqa: E402
from client_logic import get_logic                   # noqa: E402

SEEDS = (42, 7, 123)
MG_POWERS = {"per person": 0, "√MG": 0.5, "kg (MG)": 1}
SHORTAGE_WEIGHT = 1.5                 # cost of a kg short ÷ cost of a kg wasted (business decision)
SHORTAGE_WEIGHTS = (1.0, SHORTAGE_WEIGHT, 2.0)   # the decision weight, plus two for comparison
MIN_GAIN = 4            # dishes; one standard error on ~170 dishes is ~6
COST_GAIN = 0.03        # kitchen cost at least 3% lower, under every MG scaling
RMSE_TOL = 1.05

# ═══ TUNING: each value is tried on its own against production ═══
SPACE = {
    "early_stopping": [3, 5, 8],
    "dropout":        [0.1, 0.2, 0.3, 0.4, 0.5],
    "item_emb_dim":   [4, 8, 16],
    "learning_rate":  [0.0003, 0.001, 0.003, 0.01],
    "dense_units":    [[32, 16], [64, 32], [128, 64]],
    "batch":          [16, 32, 64],
    "loss":           ["mse", "mae", "huber", "pseudo_huber"],
    "loss_delta":     [0.0025, 0.005, 0.01, 0.02, 0.05],
    "optimizer":      ["adam", "adamw", "rmsprop"],
    "activation":     ["relu", "leaky_relu", "elu", "gelu", "tanh"],
    "batch_norm":     [False, True],
    "mask_padding":   [False, True],
    "extra_features": [[], ["month", "month_part"], ["mg_band"], ["month", "month_part", "mg_band"]],
    "weight_by_mg":   [False, True],
    "calibrate":      [False, True],
    "rebase":         [False, True],
}
# Settings that only make sense together, tried as one change.
LINKED = {"target": [{"target_mode": "raw", "output_activation": "linear"},
                     {"target_mode": "raw", "output_activation": "softplus"},
                     {"target_mode": "category_offset", "output_activation": "linear"}]}


def mg_cost(pred, actual, mg, weight, power=1):
    """Σ MG^power × (weight × kg/person short + kg/person over)."""
    pred, actual, mg = (np.asarray(v, dtype=float) for v in (pred, actual, mg))
    return float(np.sum(mg ** power * (weight * np.clip(actual - pred, 0, None) + np.clip(pred - actual, 0, None))))


def keeps(cand, base):
    """The keep rule: more dishes within 5 g, none lost within ±10%, lower cost at the
    decision shortage weight under every MG scaling, internal error not clearly worse."""
    costs = [f"cost {k} @{SHORTAGE_WEIGHT}" for k in MG_POWERS]
    return (cand["w5"] >= base["w5"] + MIN_GAIN and cand["within"] >= base["within"]
            and all(cand[c] <= base[c] * (1 - COST_GAIN) for c in costs)
            and cand["rmse"] <= base["rmse"] * RMSE_TOL)


class Search:
    def __init__(self, ck):
        self.ck, self.L = ck, get_logic(ck)
        dataset = test_model.CLIENT_DB[ck]["dataset"]
        c = pd.read_excel(ml_core.test_workbook_path(ck), sheet_name="cases")
        self.cases = c.assign(test=c["test"].astype(str) if "test" in c else "all")
        labels = sorted(self.cases["test"].unique())
        self.sealed = labels[-1] if len(labels) > 1 else None
        self.choose = [t for t in labels if t != self.sealed]
        self.actual = self.cases["ideal_pp"].astype(float).map(ml_core.round005).to_numpy()
        self.mg = pd.to_numeric(self.cases.get("client_mg", 1), errors="coerce").fillna(1).to_numpy(dtype=float)
        self.real = pd.to_numeric(self.cases.get("wastage", 0), errors="coerce").fillna(0).to_numpy() > 0
        self.dates = pd.to_datetime(self.cases["date"]).to_numpy()
        days = sorted(pd.to_datetime(self.cases["date"]).unique())
        own = self.cases.assign(date=pd.to_datetime(self.cases["date"])).groupby("date")["menu_items"].apply(
            lambda s: [str(x) for x in s])
        self.swapped = {d: own[days[(i + len(days) // 2) % len(days)]] for i, d in enumerate(days)}
        std = ml_core.standards_path(ck)
        self.data_key = [ml_core.file_sha(dataset), ml_core.file_sha(std) if os.path.exists(std) else "",
                         hashlib.sha256(self.cases.to_csv(index=False).encode()).hexdigest()[:12]]
        self.log = os.path.join(test_model.REPORT_DIR, ck, "search_runs.jsonl")
        os.makedirs(os.path.dirname(self.log), exist_ok=True)
        self.done = {}
        if os.path.exists(self.log):
            for line in open(self.log, encoding="utf-8"):
                r = json.loads(line)
                self.done[r["key"]] = r

    def run(self, params, seed):
        """Train once (temporary folder, production untouched); keep every test dish's prediction."""
        params = {**params, "seed": seed}
        key = json.dumps([params, self.data_key], sort_keys=True)
        if key in self.done:
            return self.done[key]
        ml_core.validate_params(params, "search")
        old, tmp = ml_core.ARTIFACT_DIR, tempfile.mkdtemp(prefix=f"{self.ck}_search_")
        preds = lambda rows: [r.get("pred") for r in rows]
        try:
            ml_core.ARTIFACT_DIR = tmp
            ml_core._model_cache.clear()
            _, rmse = ml_core.train_model(self.ck, test_model.CLIENT_DB[self.ck]["dataset"], self.L, params=params)
            rec = {"key": key, "rmse": rmse,
                   "model": preds(test_model.run_cases(self.ck, self.L, self.cases)),
                   "menus swapped": preds(test_model.run_cases(self.ck, self.L, self.cases, self.swapped))}
            offsets = ml_core.load_offsets(self.ck)
            if offsets is not None:
                rec["category average"] = [ml_core.round005(ml_core.category_offset(
                    offsets, self.L.canonicalize_category(c))) for c in self.cases["category"]]
        finally:
            ml_core.ARTIFACT_DIR = old
            ml_core._model_cache.clear()
            shutil.rmtree(tmp, ignore_errors=True)
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.done[key] = rec
        return rec

    def measure(self, pred, rows):
        """within ±10%, real-demand within, and the kitchen cost for every shortage weight and MG scaling."""
        p = np.array([0.0 if v is None else v for v in pred])[rows]   # unpredicted dish = all short
        a, mg = self.actual[rows], self.mg[rows]
        within = np.round(np.abs(p - a) / a * 100, 6) <= test_model.WITHIN_PCT   # same rules as run_cases
        w5 = np.round(np.abs(p - a) * 1000, 6) <= test_model.WITHIN_G
        day_share = pd.Series(w5).groupby(self.dates[rows]).mean()
        out = {"n": int(rows.sum()), "w5": int(w5.sum()), "within": int(within.sum()),
               "honest": int(within[self.real[rows]].sum()),
               "days": int((day_share > test_model.DAY_TARGET).sum()), "n_days": len(day_share)}
        for w in SHORTAGE_WEIGHTS:
            for k, pw in MG_POWERS.items():
                out[f"cost {k} @{w}"] = mg_cost(p, a, mg, w, pw)
        return out

    def score(self, params, tests=None, which="model"):
        """Mean over SEEDS on `tests` (default: the tests used to choose)."""
        rows = self.cases["test"].isin(tests or self.choose).to_numpy()
        recs = [self.run(params, s) for s in SEEDS]
        ms = [self.measure(r[which], rows) for r in recs]
        out = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        out["rmse"] = float(np.mean([r["rmse"] for r in recs]))
        out["seeds"] = "/".join(str(m["w5"]) for m in ms)
        return out


def row(name, s, rmse=True):
    costs = " | ".join(f"{s[f'cost {k} @{SHORTAGE_WEIGHT}']:.2f}" for k in MG_POWERS)
    rm = f"{s['rmse']:.5f}" if rmse else "—"
    return (f"| {name} | {s['w5']:.1f} / {s['n']:.0f} ({s['w5'] / s['n']:.0%}) | {s['seeds']} | "
            f"{s['days']:.1f} / {s['n_days']:.0f} | {s['within']:.1f} ({s['within'] / s['n']:.0%}) | {costs} | "
            f"{s['honest']:.1f} | {rm} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--client", required=True, help="Client key, e.g. clario.")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    S = Search(args.client)
    prod = ml_core.get_params(args.client)
    head = (f"| change | within 5 g | per seed | days >{test_model.DAY_TARGET:.0%} within 5 g | within ±10% | "
            + " | ".join(f"cost {k}" for k in MG_POWERS)
            + " | real-demand within ±10% | internal RMSE |\n|---|---|---|---|---|" + "---|" * len(MG_POWERS) + "---|---|")
    out = [f"# Model search — {args.client} ({datetime.now():%Y-%m-%d %H:%M})\n",
           f"Choosing on: {', '.join(S.choose)}. Sealed until the end: {S.sealed or 'none'}. Seeds {SEEDS}. "
           f"Costs at shortage weight {SHORTAGE_WEIGHT}. Keep rule: ≥{MIN_GAIN} more dishes within "
           f"{test_model.WITHIN_G:.0f} g AND no fewer within ±{test_model.WITHIN_PCT:.0f}% AND ≥{COST_GAIN:.0%} lower "
           f"cost under every MG scaling AND internal RMSE ≤ {RMSE_TOL:.2f}× production.\n"]

    def show(text):
        print(text, flush=True)
        out.append(text)

    name_of = lambda over: ", ".join(f"{x}={y}" for x, y in over.items())
    kg = f"cost kg (MG) @{SHORTAGE_WEIGHT}"
    base = S.score(prod)
    show("## 1. Each setting on its own (vs production)\n")
    show(head + "\n" + row("**production**", base))
    winners = {}
    candidates = [(k, {k: v}) for k, vals in SPACE.items() for v in vals if v != prod[k]]
    candidates += [(k, o) for k, opts in LINKED.items() for o in opts if any(prod[x] != y for x, y in o.items())]
    for k, over in candidates:
        s = S.score({**prod, **over})
        won = keeps(s, base)
        show(row(("✓ " if won else "") + name_of(over), s))
        if won and (k not in winners or (s["w5"], -s[kg]) > (winners[k][1]["w5"], -winners[k][1][kg])):
            winners[k] = (over, s)

    show(f"\n## 2. Combinations of the winners ({len(winners)}: {', '.join(winners) or 'none'})\n")
    kept = {name_of(o): (o, s) for o, s in winners.values()}
    if len(winners) > 1:
        show(head)
        for r in range(2, len(winners) + 1):
            for combo in itertools.combinations(winners.values(), r):
                over = {x: y for o, _ in combo for x, y in o.items()}
                s = S.score({**prod, **over})
                show(row(("✓ " if keeps(s, base) else "") + " + ".join(name_of(o) for o, _ in combo), s))
                if keeps(s, base):
                    kept[" + ".join(name_of(o) for o, _ in combo)] = (over, s)
    best_name, (best_over, _) = max(kept.items(), key=lambda kv: (kv[1][1]["w5"], -kv[1][1][kg]),
                                    default=("production (nothing passed)", ({}, base)))
    best = {**prod, **best_over}

    groups = ([("sealed " + S.sealed, [S.sealed])] if S.sealed else []) + [("all", S.choose + ([S.sealed] if S.sealed else []))]
    show(f"\n## 3. Final check — {best_name}\n")
    show(head.replace("| change |", "| model | tests |").replace("|---|---|---|", "|---|---|---|---|", 1))
    for label, params, which in [("production", prod, "model"), ("best", best, "model"),
                                 ("best, menus swapped", best, "menus swapped"),
                                 ("category average (no ML)", prod, "category average")]:
        for tl, ts in groups:
            show(row(f"{label} | {tl}", S.score(params, ts, which), rmse=which == "model"))
    show("\nKitchen cost (kg) on all tests if a shortage is weighted differently:\n")
    show("| shortage weight | production | best | category average |\n|---|---|---|---|")
    for w in SHORTAGE_WEIGHTS:
        c = [S.score(p, groups[-1][1], which)[f"cost kg (MG) @{w}"] for p, which in
             ((prod, "model"), (best, "model"), (prod, "category average"))]
        show(f"| {w:.2f}{' (decision)' if w == SHORTAGE_WEIGHT else ''} | " + " | ".join(f"{v:.1f}" for v in c) + " |")
    show(f"\nSettings to put in model_params.json: {json.dumps(best_over) if best_over else 'none — keep production'}")
    path = os.path.join(test_model.REPORT_DIR, args.client, f"search-{datetime.now():%Y%m%d-%H%M%S}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
