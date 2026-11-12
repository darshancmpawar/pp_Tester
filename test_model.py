"""
Model evaluation report for the per-pax (ideal_pp, kg/person) Keras model.

Run after every training or prediction run. Produces a Markdown report with
charts under reports/<client>/<timestamp>/ and appends a summary to
reports/<client>/runs.json, which is what "vs previous run" compares against.

Every client has a test workbook, test_cases/<client>_test.xlsx, with two tabs:
  cases     test days in the dataset's own format (date, menu_items, category,
            sub_category, ideal_pp, optional day_type / holiday_type / meal_day).
            These dates are NEVER used for training,
            tuning or evaluation (ml_core.prepare_dataset drops them).
  settings  every model setting as a row, one column per config (Config-1, Config-2 ...).
If the workbook is missing it is created from the dataset's last N_DAYS (15) days.

Usage:
    python test_model.py --client tekion              # evaluate current artifacts
    python test_model.py --client tekion --train      # retrain, then evaluate
    python test_model.py --client all                 # every embedding client
    python test_model.py --client clario --import-cases my_days.xlsx
    python test_model.py --client clario --tests      # one model per settings column
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("ARTIFACT_DIR", os.path.join(_ROOT, "artifacts"))

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

import ml_core                                               # noqa: E402
from client_database import CLIENT_DB                        # noqa: E402
from client_logic import get_logic                           # noqa: E402

REPORT_DIR = os.path.join(_ROOT, "reports")
N_DAYS     = 15   # days put on a new workbook's cases tab

# ═══ TUNING: report thresholds ════════════════════════════════════════════════
BIAS_TOL       = 0.005   # |bias| ≤ one 5 g serving-spoon step counts as "near zero"
CAT_REGRESS    = 0.0005  # category RMSE rise (kg) that counts as a regression
WITHIN_PCT     = 10.0    # a case is "within tolerance" at ±10 % error
WITHIN_G       = 5.0     # the kitchen's target: predicted within 5 g/person (one serving-spoon step) of eaten
DAY_TARGET     = 0.70    # ... for MORE than 70% of the day's dishes (10 dishes → 8 or more)

SETTING_NOTES = {
    "label":         "free-text name for this test",
    "epochs":        "training passes over the data",
    "batch":         "rows per gradient step",
    "learning_rate": "optimizer step size",
    "optimizer":     "adam | adamw | rmsprop | sgd",
    "loss":          "error the model minimises: mse | mae | huber | pseudo_huber",
    "activation":    "hidden-layer activation: relu | leaky_relu | elu | gelu | tanh",
    "loss_delta":    "huber / pseudo_huber switch point in kg: smaller = more like MAE (more dishes close)",
    "dense_units":   "neurons in hidden layer 1, layer 2 (e.g. 128,64)",
    "dropout":       "fraction of layer-1 outputs dropped while training",
    "target_mode":   "raw = predict ideal_pp | category_offset = predict gap from category mean",
    "mask_padding":  "TRUE = ignore empty co-menu slots when averaging the day's menu",
    "calibrate":     "TRUE = add each category's typical miss on the newest (validation) days to its predictions",
    "rebase":        "TRUE = scale past rows to today's standard portions (data/<client>_standard_portions.csv)",
    "early_stopping": "always on: stop after N epochs without improvement, keep the best (epochs = maximum)",
    "ensemble":      "how many models (different random starts) are trained and averaged; 1 = a single model",
    "item_emb_dim":  "embedding width for the dish and its co-menu",
    "val_split":     "fraction of training rows watched as validation",
    "test_size":     "fraction of dataset days held out as the internal test set",
}


# ── Small helpers ─────────────────────────────────────────────────────────────

def f4(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"

def f1(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"

def delta(cur, prev, fmt=f4):
    if prev is None:
        return "—"
    d = cur - prev
    return ("+" if d > 0 else "") + fmt(d)

def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)

def is_blank(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == ""

def metrics(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    err = pred - actual
    nz  = actual > 0
    return {
        "n":    int(len(actual)),
        "rmse": float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan"),
        "mae":  float(np.mean(np.abs(err))) if len(err) else float("nan"),
        "mape": float(np.mean(np.abs(err[nz]) / actual[nz]) * 100) if nz.any() else float("nan"),
        "bias": float(np.mean(err)) if len(err) else float("nan"),
    }


# ── Run history (what "previous run" means) ───────────────────────────────────

def load_runs(ck):
    p = os.path.join(REPORT_DIR, ck, "runs.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def save_runs(ck, runs):
    p = os.path.join(REPORT_DIR, ck, "runs.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)


# ── Section 1: learning curves ────────────────────────────────────────────────

def fit_history(ck):
    """Per-epoch loss / val_loss of the client's current model, or None."""
    path = ml_core._ap(ck, "fit_history.pkl")
    if not os.path.exists(path):
        return None
    hist = ml_core.joblib.load(path)
    return hist if hist.get("loss") and hist.get("val_loss") else None


def best_epoch_label(best, n):
    """'best epoch 10', or 'lowest so far (last epoch)' when the curve never bottomed out."""
    return "lowest so far (last epoch)" if best == n else f"best epoch {best}"


def fit_diagnosis(hist):
    """
    (verdict, explanation) from a fit history: verdict is one of
    "overfitting", "underfitting", "still learning", "good fit".

    target_std is the spread of what the model was trained on (ideal_pp, or the
    gap from the category mean for category_offset). A validation RMSE close to
    it means the model barely beats predicting that mean for every dish.
    """
    loss, val = hist["loss"], hist["val_loss"]
    n, best = len(val), int(np.argmin(val)) + 1
    # the recorded RMSE at the kept epoch; older models only have the loss (its root is an RMSE for mse)
    val_rmse = float(hist["val_rmse"][best - 1]) if "val_rmse" in hist else float(np.sqrt(min(val)))
    target_std = hist["target_std"]
    if hist.get("early_stopping") and n < hist["max_epochs"]:
        if val_rmse > 0.8 * target_std:
            return "underfitting", (
                f"Early stopping kept epoch {best}, but its validation RMSE {val_rmse * 1000:.1f} g is close to "
                f"the spread of the target ({target_std * 1000:.1f} g): the model barely beats a constant.")
        return "good fit", (
            f"Early stopping ended training at epoch {n} and kept the weights from epoch {best}, "
            f"where validation loss was lowest. Best validation RMSE {val_rmse * 1000:.1f} g vs target "
            f"spread {target_std * 1000:.1f} g.")
    if best <= n - 3 and val[-1] > min(val) * 1.10 and loss[-1] < loss[best - 1]:
        return "overfitting", (
            f"Validation loss bottomed at epoch {best} and rose {(val[-1] / min(val) - 1) * 100:.0f}% "
            f"by epoch {n} while training loss kept falling: the model is memorising the "
            f"training days. Lower epochs to about {best}, or raise dropout.")
    if val_rmse > 0.8 * target_std:
        return "underfitting", (
            f"Best validation RMSE {val_rmse * 1000:.1f} g is close to the spread of the target "
            f"({target_std * 1000:.1f} g): the model barely beats a constant. Try more epochs, a higher "
            f"learning rate or wider layers.")
    if best == n and n >= 3 and val[-1] < val[-3] * 0.98:
        return "still learning", (
            f"Validation loss was still falling when training stopped at epoch {n}, so its lowest "
            f"point is simply the last epoch; it has not bottomed out. More epochs may help.")
    return "good fit", (
        f"Validation loss levelled off without rising; best epoch {best} of {n}. "
        f"Best validation RMSE {val_rmse * 1000:.1f} g vs target spread {target_std * 1000:.1f} g.")


def learning_curves(ck, out_dir):
    hist = fit_history(ck)
    if hist is None:
        return None, "No fit history saved for this model. Retrain it to get learning curves."
    loss, val = hist["loss"], hist["val_loss"]
    epochs = np.arange(1, len(loss) + 1)
    best = int(np.argmin(val)) + 1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, loss, marker="o", ms=3, label="train loss (MSE)")
    ax.plot(epochs, val, marker="o", ms=3, label="validation loss (MSE)")
    ax.axvline(best, color="grey", ls="--", lw=1, label=best_epoch_label(best, len(val)))
    ax.set_yscale("log")   # epoch-1 loss is ~100× the rest; linear hides the tail
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (kg/person)², log scale")
    ax.set_title(f"{ck} — learning curves"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=110)
    plt.close(fig)
    verdict, text = fit_diagnosis(hist)
    return {"best_epoch": best, "epochs": len(val)}, f"**{verdict.capitalize()}** — {text}"


# ── Sections 2–4: held-out test set ───────────────────────────────────────────

def evaluate_test_set(ck, info, L):
    df = ml_core.prepare_dataset(ck, info["dataset"], L)
    if ml_core.get_params(ck)["rebase"]:   # judged at today's standards, as it was trained
        df = ml_core.rebase_to_standards(df, pd.read_csv(ml_core.standards_path(ck)))
    columns = ml_core.load_map(ck, "features")
    encoders = {c: ml_core.load_enc(ck, c) for c in columns}
    try:
        X, y = ml_core.build_inputs(df, columns, encoders)
    except ValueError as e:
        raise RuntimeError(
            f"[{ck}] the dataset contains values the saved encoders have never seen "
            f"({e}). The dataset changed since training — run with --train."
        ) from e
    params = ml_core.get_params(ck)
    train_idx, test_idx = ml_core.split_train_test(df, params["test_size"])
    models = ml_core.load_models(ck)
    offsets, calibration = ml_core.load_offsets(ck), ml_core.load_calibration(ck)

    def predicted(idx):
        """pred = the average, as predict() serves it; pred_1, pred_2, … = each model alone,
        with the same offset and calibration; calibration kept to take it back out."""
        rows = df.iloc[idx].copy()
        rows["actual"] = y[idx]
        offset = (rows["category"].map(lambda c: ml_core.category_offset(offsets, c)).to_numpy(dtype=float)
                  if offsets is not None else 0.0)   # category_offset model predicts the gap from the category mean
        rows["calibration"] = rows["category"].map(lambda c: (calibration or {}).get(ml_core.token(c), 0.0))
        shift = offset + rows["calibration"].to_numpy()
        raw = [m.predict([a[idx] for a in X], verbose=0).flatten() for m in models]
        for i, p in enumerate(raw, 1):
            rows[f"pred_{i}"] = p + shift
        rows["pred"] = np.mean(raw, axis=0) + shift   # as predict() serves it: average, then shift
        return rows

    test = predicted(test_idx)
    # Keras validation_split holds out the LAST share of the training rows; early stopping watches them.
    val = predicted(train_idx[int(len(train_idx) * (1 - params["val_split"])):])

    # Dumb baseline: mean ideal_pp of the dish's category in the TRAIN rows.
    train = df.iloc[train_idx]
    cat_mean = train.groupby("category")["ideal_pp"].mean()
    test["baseline"] = test["category"].map(cat_mean).fillna(train["ideal_pp"].mean())
    return df, train, test, val


def shortage_rows(frame):
    """
    (dishes scored, the dishes predicted more than WITHIN_PCT below what was eaten), both on the
    app's 5 g grid, worst first. kg short = (eaten − predicted) per person × client MG.
    """
    f = frame.rename(columns={"menu_items": "dish", "client_mg": "mg"})
    f = f[f["pred"].notna()]
    eaten, pred = f["actual"].map(ml_core.round005), f["pred"].astype(float).map(ml_core.round005)
    short = ((eaten - pred) / eaten * 100).round(6) > WITHIN_PCT
    out = f.loc[short, ["date", "dish", "category", "mg"]].assign(
        predicted_g=pred[short] * 1000, eaten_g=eaten[short] * 1000)
    out["short_g"] = out["eaten_g"] - out["predicted_g"]
    out["kg_short"] = out["short_g"] / 1000 * pd.to_numeric(out["mg"], errors="coerce")
    return len(f), out.sort_values("kg_short", ascending=False)


def efficiency(name, frame):
    """Every measure for one set of dishes, on the app's 5 g grid (kg/person columns actual, pred;
    client MG in mg or client_mg). A dish that could not be predicted counts as a miss."""
    f = frame[frame["pred"].notna()]
    a = f["actual"].astype(float).map(ml_core.round005).to_numpy()
    d = f["pred"].astype(float).map(ml_core.round005).to_numpy() - a
    grams = np.round(np.abs(d) * 1000, 6)   # round(): 5 g steps come out as 5.0000001
    w5 = pd.Series(grams <= WITHIN_G, index=f.index)
    w10 = np.round(np.abs(d) / np.where(a > 0, a, np.nan) * 100, 6) <= WITHIN_PCT
    mg = pd.to_numeric(f["mg"] if "mg" in f else f["client_mg"], errors="coerce").fillna(0).to_numpy()
    day_share = w5.reindex(frame.index, fill_value=False).groupby(pd.to_datetime(frame["date"])).mean()
    _, short = shortage_rows(frame)
    n = len(frame)
    return {"Set": name, "Dishes": n, "Days": len(day_share),
            "Within 5 g (%)": 100 * w5.sum() / n, "Exact (%)": 100 * (grams == 0).sum() / n,
            "Within ±10% (%)": 100 * w10.sum() / n, "Days over 70%": int((day_share > DAY_TARGET).sum()),
            "RMSE (g)": 1000 * float(np.sqrt(np.mean(d ** 2))), "MAE (g)": 1000 * float(np.mean(np.abs(d))),
            "Bias (g)": 1000 * float(np.mean(d)), "Shortages": len(short),
            "kg short": float(short["kg_short"].sum()), "kg over": float(np.sum(np.clip(d, 0, None) * mg))}


def evaluate_current(ck):
    """What the app's Model Test page shows for the model on disk (no files written): every
    test-case dish, and every measure for the validation rows, the internal test set, each
    test set, all of them, and the category average (no model) on all of them."""
    L = get_logic(ck)
    _, train, test, val = evaluate_test_set(ck, CLIENT_DB[ck], L)
    cases = run_cases(ck, L, read_frame(ensure_workbook(ck, L)[0], "cases"))
    c = pd.DataFrame(cases)
    cat_mean = train.groupby("category")["ideal_pp"].mean()
    no_model = c.assign(pred=c["category"].map(L.canonicalize_category).map(cat_mean)
                        .fillna(train["ideal_pp"].mean()))
    sets = [("Validation (newest training days)", val), ("Internal test", test)]
    sets += [(str(t), g) for t, g in c.groupby("test")] if c["test"].notna().any() else []
    sets += [("All test cases", c), ("All test cases — category average, no model", no_model)]
    # Each averaged model alone, scored exactly like the overall table (so the average's row equals it).
    n = len(ml_core.load_models(ck))
    columns = [(f"Model {i}", f"pred_{i}") for i in range(1, n + 1)] + ([(f"Average of {n} (used)", "pred")] if n > 1 else [])

    def scored(name, col):
        v, t = efficiency(name, val.assign(pred=val[col])), efficiency(name, test.assign(pred=test[col]))
        return {"Model": name, "Validation RMSE (g)": v["RMSE (g)"], "Internal test RMSE (g)": t["RMSE (g)"],
                "Internal test within 5 g (%)": t["Within 5 g (%)"]}
    models = [scored(name, col) for name, col in columns]
    # The learning curve's number, recomputed now: model 1 on the validation rows before calibration,
    # as Keras measured it during training (kg/person). It must match the curve at the kept epoch.
    curve_check = float(np.sqrt(np.mean((val["pred_1"] - val["calibration"] - val["actual"]) ** 2)))
    return {"cases": cases, "overall": [efficiency(name, frame) for name, frame in sets],
            "models": models, "curve_check": curve_check}


def data_study(ck):
    """(the training data by year, each category's cooked and eaten portion by year) — the raw
    dataset, before re-basing. Eaten = (cooked − wastage) ÷ client MG."""
    L = get_logic(ck)
    df = ml_core.load_dataset(ck, CLIENT_DB[ck]["dataset"], L)
    df["year"] = df["date"].dt.year
    day = df.groupby("date").agg(year=("year", "first"), dishes=("menu_items", "size"),
                                 mg=("client_mg", "first"), kind=("category", ml_core.day_kind))
    days, dishes = day.groupby("year"), df.groupby("year")
    dates = day.index.to_series().groupby(day["year"])
    rows = {
        "First – last day": dates.agg(lambda d: f"{d.min():%d %b} – {d.max():%d %b}"),
        "Days logged": days.size(),
        "Dishes logged": dishes.size(),
        "Dishes per day": days["dishes"].mean().round(1),
        "Different dishes": dishes["menu_items"].nunique(),
        "Average client MG": days["mg"].mean().round(0),
        "Dishes with zero wastage, %": (pd.to_numeric(df["wastage"], errors="coerce").eq(0)
                                        .groupby(df["year"]).mean() * 100).round(0),
        "Fridays": dates.agg(lambda d: int((d.dt.day_name() == "Friday").sum())),
        **{f"{k.capitalize()} days": day["kind"].eq(k).groupby(day["year"]).sum() for k in sorted(day["kind"].unique())},
    }
    summary = pd.DataFrame(rows).T.astype(str).rename_axis("Measure").reset_index()
    grams = lambda col: (df.groupby(["category", "year"])[col].median() * 1000).round(0).unstack()
    portions = pd.concat([grams("per_pax").add_prefix("Cooked "), grams("ideal_pp").add_prefix("Eaten ")], axis=1)
    path = ml_core.standards_path(ck)
    if os.path.exists(path):
        std = pd.read_csv(path)
        std = std[std["day"].isna()].assign(category=lambda s: s["category"].map(ml_core.norm))
        portions.insert(0, "Standard today", std.set_index("category")["grams"])
    portions.columns = [f"{c} (g)" for c in portions.columns]
    return summary, portions.rename_axis("Category").reset_index()


def scatter_and_residuals(ck, test, out_dir):
    a, p = test["actual"].to_numpy(), test["pred"].to_numpy()
    res = p - a
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.scatter(a, p, s=12, alpha=0.6)
    lo, hi = float(min(a.min(), p.min())), float(max(a.max(), p.max()))
    ax1.plot([lo, hi], [lo, hi], color="red", lw=1, label="y = x (perfect)")
    ax1.set_xlabel("actual per-pax (kg)"); ax1.set_ylabel("predicted per-pax (kg)")
    ax1.set_title("Predicted vs actual (test set)"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.scatter(p, res, s=12, alpha=0.6)
    ax2.axhline(0, color="red", lw=1)
    ax2.set_xlabel("predicted per-pax (kg)"); ax2.set_ylabel("error = predicted − actual (kg)")
    ax2.set_title("Residuals"); ax2.grid(alpha=0.3)
    fig.suptitle(ck); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pred_vs_actual.png"), dpi=110)
    plt.close(fig)

    # Pattern note: mean residual in low / mid / high thirds of the prediction.
    notes = []
    if len(p) >= 9:
        q1, q2 = np.quantile(p, [1 / 3, 2 / 3])
        for name, m in (("small", p <= q1), ("medium", (p > q1) & (p <= q2)), ("large", p > q2)):
            mr = float(res[m].mean())
            if abs(mr) > BIAS_TOL:
                notes.append(f"{'over' if mr > 0 else 'under'}-predicts {name} portions "
                             f"by {abs(mr):.4f} kg on average")
        slope = float(np.polyfit(a, p, 1)[0]) if np.ptp(a) > 0 else 1.0
        if slope < 0.8:
            notes.append(f"predictions are compressed toward the mean (fit slope {slope:.2f} "
                         "< 1): big portions come out too small and small ones too big")
    # Small categories (e.g. biryani) can be biased without moving the thirds.
    for cat, g in test.groupby("category"):
        mr = float((g["pred"] - g["actual"]).mean())
        if abs(mr) > BIAS_TOL:
            notes.append(f"'{cat}' is {'over' if mr > 0 else 'under'}-predicted by "
                         f"{abs(mr):.4f} kg on average ({len(g)} rows)")
    return "; ".join(notes) + "." if notes else (
        f"No systematic over/under pattern (every third and category within ±{BIAS_TOL} kg).")


# ── Test workbook (cases + settings tabs) ─────────────────────────────────────

def read_frame(path, sheet=0):
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def fixed_settings(ck, L):
    """Settings that are shared, not per test — listed for reference only."""
    feats = [c for c in L.encoder_columns if c != "menu_items"]
    return [
        ("architecture", "(fixed) dish emb + mean co-menu emb + one emb per feature "
                         "→ Dense → Dropout → Dense → Dense(1)"),
        ("output_layer", "(fixed) Dense(1), linear — predicts ideal_pp in kg/person"),
        ("features",     "(fixed) " + ", ".join(L.encoder_columns)
                         + ("; meal_day = veg / the day's non-veg category, from the menu"
                            if getattr(L, "meal_day_from_menu", False) else "")),
        ("feature_emb_dims", "(fixed) " + ", ".join(f"{c}={ml_core.EMB_DIMS.get(c, 2)}" for c in feats)),
        ("ctx_len",      f"(fixed) {ml_core.CTX_LEN} co-menu dishes per row"),
        ("rounding",     "(fixed) test-case predictions and actuals rounded to the nearest 0.005 kg (5 g)"),
        ("holdout",      "(fixed) every date on the cases tab is removed from training data"),
    ]


def settings_frame(ck, L, tests):
    """tests: list of (column name, label, params)."""
    rows = []
    for key in ["label"] + list(ml_core.DEFAULT_PARAMS):
        row = {"parameter": key, "description": SETTING_NOTES.get(key, "")}
        for name, label, params in tests:
            v = label if key == "label" else params[key]
            row[name] = ",".join(str(u) for u in v) if isinstance(v, list) else v
        rows.append(row)
    for key, desc in fixed_settings(ck, L):
        rows.append({"parameter": key, "description": desc, **{n: "" for n, _, _ in tests}})
    return pd.DataFrame(rows)


def write_workbook(path, cases, settings):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        cases.to_excel(xw, sheet_name="cases", index=False)
        settings.to_excel(xw, sheet_name="settings", index=False)
        for ws in xw.sheets.values():
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 60)


def default_tests(ck):
    tuned = ml_core.get_params(ck)
    is_tuned = tuned != ml_core.DEFAULT_PARAMS
    tests = [("Config-1", "tuned (model_params.json)" if is_tuned else "defaults", tuned)]
    if is_tuned:
        tests.append(("Config-2", "defaults", dict(ml_core.DEFAULT_PARAMS)))
    return tests


def ensure_workbook(ck, L):
    """Create test_cases/<ck>_test.xlsx from the dataset's last N_DAYS if missing."""
    path = ml_core.test_workbook_path(ck)
    if os.path.exists(path):
        return path, False
    raw = pd.read_excel(CLIENT_DB[ck]["dataset"], sheet_name=CLIENT_DB[ck].get("sheet", 0))
    dcol = [c for c in raw.columns if str(c).strip().lower() == "date"][0]
    dates = pd.to_datetime(raw[dcol], errors="coerce").dt.normalize()
    last = sorted(dates.dropna().unique())[-N_DAYS:]
    write_workbook(path, raw[dates.isin(last)], settings_frame(ck, L, default_tests(ck)))
    return path, True


def import_cases(ck, L, src):
    """Replace the cases tab with the first sheet of src; keep existing settings."""
    path = ml_core.test_workbook_path(ck)
    cases = pd.read_excel(src)
    if os.path.exists(path):
        settings = pd.read_excel(path, sheet_name="settings")
    else:
        settings = settings_frame(ck, L, default_tests(ck))
    write_workbook(path, cases, settings)
    return path


def parse_settings(path):
    """[(column name, label, params)] for every 'Config-…' column on the settings tab."""
    sheet = pd.read_excel(path, sheet_name="settings").set_index("parameter")
    tests = []
    for name in [c for c in sheet.columns if str(c).lower().startswith("config")]:
        params = dict(ml_core.DEFAULT_PARAMS)
        for key in ml_core.DEFAULT_PARAMS:
            if key not in sheet.index or is_blank(sheet.at[key, name]):
                continue
            v = sheet.at[key, name]
            if key == "dense_units":
                params[key] = [int(float(u)) for u in str(v).split(",")]
            elif key in ("mask_padding", "calibrate", "rebase"):
                params[key] = str(v).strip().lower() in ("true", "1", "yes")
            elif key in ("optimizer", "loss", "activation", "target_mode"):
                params[key] = str(v).strip().lower()
            elif key in ("epochs", "batch", "item_emb_dim", "early_stopping", "seed", "ensemble"):
                params[key] = int(float(v))
            else:
                params[key] = float(v)
        label = sheet.at["label", name] if "label" in sheet.index else ""
        ml_core.validate_params(params, f"{os.path.basename(path)} settings [{name}]")
        tests.append((str(name), "" if is_blank(label) else str(label), params))
    return tests


def params_text(p):
    return (f"epochs={p['epochs']}, batch={p['batch']}, lr={p['learning_rate']}, "
            f"optimizer={p['optimizer']}, loss={p['loss']}, activation={p['activation']}, "
            f"layers={','.join(str(u) for u in p['dense_units'])}, dropout={p['dropout']}, "
            f"item_emb={p['item_emb_dim']}, target={p['target_mode']}, mask_padding={p['mask_padding']}, "
            f"early_stopping={p['early_stopping']}, calibrate={p['calibrate']}, rebase={p['rebase']}")


# ── Section 5: the test-case days ──────────────────────────────────────────

def run_cases(ck, L, cases, swap=None):
    """
    Predict every dish on the cases tab with the model currently in ARTIFACT_DIR.
    swap: optional {date: other date} — predict each day with the other day's menu instead
    of its own (scripts/search.py does this to check the model reads the menu).
    """
    cases = cases.copy()
    cases["date"] = pd.to_datetime(cases["date"], errors="coerce")
    menus = {pd.Timestamp(d): [str(x) for x in g["menu_items"]] for d, g in cases.groupby("date")}   # whole days
    kinds = {pd.Timestamp(d): ml_core.day_kind([L.canonicalize_category(c) for c in g["category"]])
             for d, g in cases.groupby("date")}   # veg / the non-veg category, when the model uses it

    def has(r, c):
        return c in r and not is_blank(r[c])

    out = []
    for n, c in enumerate(cases.to_dict("records"), 1):
        menu_day = (swap or {}).get(pd.Timestamp(c["date"]), pd.Timestamp(c["date"]))
        res = ml_core.predict(   # the day's real headcount, as the app passes it
            ck, str(c["menu_items"]), menus[menu_day], int(c["client_mg"]) if has(c, "client_mg") else 1,
            c.get("sub_category"), c.get("category"), c["date"].day_name(), L,
            day_type=c["day_type"] if has(c, "day_type") else None,
            holiday_type=c["holiday_type"] if has(c, "holiday_type") else None,
            meal_day=(kinds[pd.Timestamp(c["date"])] if getattr(L, "meal_day_from_menu", False)
                      else c["meal_day"] if has(c, "meal_day") else None),
            service_date=c["date"],
        )
        actual = ml_core.round005(float(c["ideal_pp"]))   # same 5 g grid as the prediction
        row = {"case": n, "test": c.get("test"), "mg": c.get("client_mg"), "date": c["date"].date(),
               "kind": kinds[pd.Timestamp(c["date"])], "dish": c["menu_items"],
               "category": c["category"], "actual": actual,
               "error": res.error, "fallback": res.fallback_item if res.fallback else None,
               "within": False, "acc": 0.0,   # a dish that cannot be predicted counts as a miss
               "within_5g": False, "exact": False}
        if res.ok:
            pred = res.per_pax
            diff = pred - actual
            pct = diff / actual * 100 if actual > 0 else float("nan")
            # round(): on the 5 g grid exactly-10% misses (90 g vs 100 g) come out as 10.0000001%
            within = bool(round(abs(pct), 6) <= WITHIN_PCT) if actual > 0 else False
            # Accuracy = dishes within ±WITHIN_PCT / all dishes, so each dish scores 100 or 0.
            row.update(pred=pred, diff=abs(diff), signed_diff=diff, signed_pct=pct,
                       pct=abs(pct), within=within, acc=100.0 if within else 0.0,
                       within_5g=bool(round(abs(diff) * 1000, 6) <= WITHIN_G),   # both on the 5 g grid
                       exact=bool(round(abs(diff) * 1000, 6) == 0))

        out.append(row)
    return out


def day_tables(case_rows):
    """Markdown: one heading + dish table per test-case day."""
    out = []
    by_day = {}
    for r in case_rows:
        by_day.setdefault(r["date"], []).append(r)
    for day, rows in sorted(by_day.items()):
        within, w5 = sum(r["within"] for r in rows), sum(r["within_5g"] for r in rows)
        out.append(f"### {day:%a %d %b %Y} — {len(rows)} dishes · {w5}/{len(rows)} within {WITHIN_G:.0f} g · "
                   f"accuracy {within / len(rows) * 100:.1f}% ({within}/{len(rows)} within ±{WITHIN_PCT:.0f}%)\n")
        table = []
        for r in rows:
            if r.get("pred") is None:
                table.append([r["dish"], r["category"], "—", f4(r["actual"]), "—", "—", "no", "no", r["error"]])
                continue
            note = f"fallback: {r['fallback']}" if r["fallback"] else ""
            table.append([r["dish"], r["category"], f4(r["pred"]), f4(r["actual"]),
                          f"{r['signed_diff']:+.4f}", f"{r['signed_pct']:+.1f}",
                          "yes" if r["within_5g"] else "no", "yes" if r["within"] else "no", note])
        out.append(md_table(["dish", "category", "predicted", "actual", "difference", "difference %",
                             f"within {WITHIN_G:.0f} g", f"within ±{WITHIN_PCT:.0f}%", "note"], table))
        out.append("")
    return "\n".join(out)


def peeked_dates(ck, L, cases):
    """Case dates that are in the dataset but were NOT held out when the model trained."""
    hist_path = ml_core._ap(ck, "fit_history.pkl")
    held = set()
    if os.path.exists(hist_path):
        held = set(ml_core.joblib.load(hist_path).get("holdout_dates", []))
    raw = ml_core.load_dataset(ck, CLIENT_DB[ck]["dataset"], L)
    data_dates = {str(d.date()) for d in pd.to_datetime(raw["date"], errors="coerce").dropna()}
    case_dates = {str(d.date()) for d in pd.to_datetime(cases["date"], errors="coerce").dropna()}
    return sorted((case_dates & data_dates) - held)


# ── Report ────────────────────────────────────────────────────────────────────

def report_client(ck, retrain=False):
    info, L = CLIENT_DB[ck], get_logic(ck)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(REPORT_DIR, ck, stamp)
    os.makedirs(out_dir, exist_ok=True)

    if retrain:
        print(f"[{ck}] training…")
        ml_core._model_cache.clear()
        _, train_rmse = ml_core.train_model(ck, info["dataset"], L, verbose=0)
        print(f"[{ck}] trained, RMSE={train_rmse:.4f}")
    elif not ml_core.artifacts_exist(ck, L.encoder_columns):
        raise RuntimeError(f"[{ck}] no trained artifacts — run with --train.")

    runs = load_runs(ck)
    prev = runs[-1] if runs else None

    # 1 — learning curves
    df, train, test, _ = evaluate_test_set(ck, info, L)
    fit, fit_diag = learning_curves(ck, out_dir)

    # 2 — test-set metrics
    m      = metrics(test["actual"], test["pred"])
    base   = metrics(test["actual"], test["baseline"])
    beats_base = m["rmse"] < base["rmse"]

    # 3 — per category
    per_cat = {}
    for cat, g in test.groupby("category"):
        per_cat[cat] = metrics(g["actual"], g["pred"])
    prev_cat = (prev or {}).get("per_category", {})
    regressed = [c for c, v in per_cat.items()
                 if c in prev_cat and v["rmse"] - prev_cat[c]["rmse"] > CAT_REGRESS]

    # 4 — scatter / residuals
    pattern = scatter_and_residuals(ck, test, out_dir)

    # 5 — cases
    cases_path, auto_cases = ensure_workbook(ck, L)
    cases = read_frame(cases_path, "cases")
    missing = [c for c in ["date", "menu_items", "category", "sub_category", "ideal_pp"] if c not in cases.columns]
    if missing:
        raise RuntimeError(f"{cases_path} cases tab is missing columns {missing}")
    peeked = peeked_dates(ck, L, cases)
    case_rows = run_cases(ck, L, cases)
    ok_rows = [r for r in case_rows if r.get("pred") is not None]
    n_days = cases["date"].nunique()

    # 6 — verdict
    checks = []
    if prev:
        pm = prev["metrics"]
        checks.append(("RMSE and MAPE both dropped",
                       m["rmse"] < pm["rmse"] and m["mape"] < pm["mape"],
                       f"RMSE {pm['rmse']:.4f} → {m['rmse']:.4f} and MAPE {pm['mape']:.1f}% → {m['mape']:.1f}% "
                       "did not both drop"))
        checks.append((f"|bias| ≤ {BIAS_TOL} kg", abs(m["bias"]) <= BIAS_TOL,
                       f"bias {m['bias']:+.4f} kg is outside ±{BIAS_TOL}"))
        checks.append(("no category got worse", not regressed,
                       f"{len(regressed)} categor{'y' if len(regressed) == 1 else 'ies'} got worse: "
                       + ", ".join(f"{c} ({prev_cat[c]['rmse']:.4f} → {per_cat[c]['rmse']:.4f})" for c in regressed)))
        improved = all(ok for _, ok, _ in checks)
        if m["rmse"] == pm["rmse"] and m["mape"] == pm["mape"]:
            # Same scores to the last digit: the model was re-tested, not changed.
            verdict = f"UNCHANGED — same model and results as run {prev['timestamp']}; nothing to accept or reject."
        elif improved:
            verdict = "ACCEPT — better than the previous run on every check."
        else:
            verdict = "REJECT — " + "; ".join(why for _, ok, why in checks if not ok) + "."
    else:
        verdict = ("FIRST RUN — baseline recorded, nothing to compare yet. Model "
                   f"{'beats' if beats_base else 'does NOT beat'} the category-mean baseline.")

    one_line = (f"{ck}: {verdict.split(' — ')[0]} · RMSE {m['rmse']:.4f} kg "
                f"({delta(m['rmse'], prev['metrics']['rmse']) if prev else 'no previous'}) · "
                f"bias {m['bias']:+.4f} · {'beats' if beats_base else 'LOSES TO'} baseline")

    # ── Write markdown ────────────────────────────────────────────────────────
    L_ = []
    w = L_.append
    w(f"# Model report — {info['name']} ({stamp})\n")
    w(f"**Verdict:** {one_line}\n")
    w(f"Mode: {'trained + evaluated' if retrain else 'evaluated existing model'} · "
      f"test rows: {m['n']} (held-out dates, {int(ml_core.get_params(ck)['test_size'] * 100)}% of days) · "
      f"previous run: {prev['timestamp'] if prev else 'none'}\n")

    w("## 1. Learning curves\n")
    if fit:
        w("![learning curves](learning_curves.png)\n")
        w(f"{fit_diag}\n")
    else:
        w(f"{fit_diag}\n")

    w("## 2. Test-set metrics (the gate)\n")
    pm = prev["metrics"] if prev else {}
    w(md_table(
        ["metric", "this run", "previous", "Δ", "category-mean baseline"],
        [["RMSE (kg/person) — primary", f4(m["rmse"]), f4(pm.get("rmse")), delta(m["rmse"], pm.get("rmse")), f4(base["rmse"])],
         ["MAE (kg/person)", f4(m["mae"]), f4(pm.get("mae")), delta(m["mae"], pm.get("mae")), f4(base["mae"])],
         ["MAPE (%)", f1(m["mape"]), f1(pm.get("mape")), delta(m["mape"], pm.get("mape"), f1), f1(base["mape"])],
         ["Bias (pred − actual)", f"{m['bias']:+.4f}", f4(pm.get("bias")), delta(m["bias"], pm.get("bias")), f"{base['bias']:+.4f}"]]))
    w("")
    w("Bias sign: **+** = over-predicts (waste), **−** = under-predicts (shortage). "
      "Metrics use the raw model output; the app and the test cases round it to the nearest 5 g.\n")
    vs_prev = ("no previous run to compare" if not prev else
               "better" if m["rmse"] < pm["rmse"] else "worse" if m["rmse"] > pm["rmse"] else "unchanged")
    vs_prev += "" if not prev else " vs previous run"
    w(f"**Verdict:** {vs_prev} · "
      f"{'better' if beats_base else 'WORSE'} than baseline "
      f"(RMSE {abs(1 - m['rmse'] / base['rmse']) * 100:.1f}% "
      f"{'lower' if beats_base else 'higher'} than baseline).\n")

    w("## 3. Per-category breakdown (worst first)\n")
    rows = []
    for cat, v in sorted(per_cat.items(), key=lambda kv: -kv[1]["rmse"]):
        pr = prev_cat.get(cat, {}).get("rmse")
        flag = "⚠ worse" if cat in regressed else ""
        rows.append([cat, v["n"], f4(v["rmse"]), delta(v["rmse"], pr), f1(v["mape"]), f"{v['bias']:+.4f}", flag])
    w(md_table(["category", "count", "RMSE", "Δ vs prev", "MAPE %", "bias", "flag"], rows))
    w("")
    w(f"Regressed categories: **{', '.join(regressed) if regressed else 'none'}**"
      f"{'' if prev else ' (no previous run)'}.\n")
    w("**By weekday**\n")
    w(md_table(["weekday", "count", "RMSE", "MAPE %", "bias"],
               [[day, v["n"], f4(v["rmse"]), f1(v["mape"]), f"{v['bias']:+.4f}"]
                for day, v in ((d, metrics(g["actual"], g["pred"])) for d, g in test.groupby("weekday"))]))
    w("")

    w("## 4. Predicted vs actual\n")
    w("![predicted vs actual](pred_vs_actual.png)\n")
    w(f"Pattern: {pattern}\n")

    w(f"## 5. Test cases ({n_days} days, {len(case_rows)} dishes)\n")
    w(f"Source: `{os.path.relpath(cases_path, _ROOT)}` (cases tab)"
      + (" — **auto-created from the dataset's last days; use `--import-cases` for your own.**" if auto_cases else "")
      + ". These dates are excluded from training.\n")
    if peeked:
        w(f"⚠ **The current model was trained on {len(peeked)} of these days** "
          f"({', '.join(peeked[:5])}{'…' if len(peeked) > 5 else ''}). Results below are not "
          f"honest — retrain (`python scripts/train.py --client {ck} --force`).\n")
    w(day_tables(case_rows))
    if ok_rows:
        cdf = pd.DataFrame(case_rows)
        g = cdf.groupby("category").agg(n=("acc", "size"), avg_diff=("diff", "mean"), avg_acc=("acc", "mean"))
        w("### All days, by category\n")
        w(md_table(["category", "cases", "avg diff (kg)", f"accuracy % (within ±{WITHIN_PCT:.0f}%)"],
                   [[c, int(v.n), f4(v.avg_diff), f1(v.avg_acc)] for c, v in g.iterrows()]))
        w("")
        within_n = int(cdf["within"].sum())
        w(f"**OVERALL:** avg diff {cdf['diff'].mean():.4f} kg · accuracy {cdf['acc'].mean():.1f}% "
          f"({within_n} of {len(case_rows)} within ±{WITHIN_PCT:.0f}%; "
          f"{len(case_rows) - len(ok_rows)} could not be predicted).\n")
        days_ok = (cdf.groupby("date")["within_5g"].mean() > DAY_TARGET)
        w(f"**Within {WITHIN_G:.0f} g (the kitchen's target):** {int(cdf['within_5g'].sum())} of {len(cdf)} dishes "
          f"({cdf['within_5g'].mean():.0%}), {int(cdf['exact'].sum())} exact · more than {DAY_TARGET:.0%} of the "
          f"day's dishes within {WITHIN_G:.0f} g on {int(days_ok.sum())} of {len(days_ok)} days.\n")
        if cdf["test"].notna().any():   # cases tab holds several test sets
            w(" · ".join(f"**{t}:** {int(g['within_5g'].sum())}/{len(g)} within {WITHIN_G:.0f} g, "
                         f"{g['acc'].mean():.1f}% within ±{WITHIN_PCT:.0f}%" for t, g in cdf.groupby("test")) + "\n")
        cases_summary = {"avg_diff": float(cdf["diff"].mean()), "avg_acc": float(cdf["acc"].mean()),
                         "within": within_n}
    else:
        w("No case could be predicted.\n")
        cases_summary = None

    w("## 6. Final verdict\n")
    for n, ok, _ in checks:
        w(f"- {'✅' if ok else '❌'} {n}")
    if checks:
        w("")
    w(f"**{verdict}**\n")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L_))

    runs.append({
        "timestamp": stamp, "mode": "train" if retrain else "evaluate",
        "metrics": {**m, "baseline_rmse": base["rmse"]},
        "per_category": per_cat, "fit": fit, "cases": cases_summary,
        "verdict": verdict, "peeked": peeked,
        "checks": [[n, ok] for n, ok, _ in checks], "previous": prev["timestamp"] if prev else None,
    })
    save_runs(ck, runs)

    print(one_line)
    print(f"   report: {report_path}")
    return report_path


# ── Settings tests: one throwaway model per settings column ──────────────────

def split_text(df, params, n_case_days):
    """Human-readable train / val / test split for one settings column."""
    train_idx, test_idx = ml_core.split_train_test(df, float(params["test_size"]))
    days = df["date"].dt.normalize()
    vs = float(params["val_split"])
    n_val = int(round(len(train_idx) * vs))
    return (f"train {len(train_idx) - n_val} rows · val {n_val} rows ({vs:.0%} of train) · "
            f"internal test {len(test_idx)} rows / {days.iloc[test_idx].nunique()} days "
            f"({float(params['test_size']):.0%}) · test cases {n_case_days} days (never trained on)")


def trial(ck, L, params, cases):
    """
    Train a throwaway model with these settings (test-case days excluded) and
    predict every test-case dish. Returns (internal test RMSE, case rows, fit
    history, internal test rows with "actual" and "pred"). Production artifacts
    are never touched.
    """
    import shutil
    import tempfile
    old_dir = ml_core.ARTIFACT_DIR
    tmp = tempfile.mkdtemp(prefix=f"{ck}_trial_")
    try:
        ml_core.ARTIFACT_DIR = tmp
        ml_core._model_cache.clear()
        _, rmse = ml_core.train_model(ck, CLIENT_DB[ck]["dataset"], L, params=params)
        _, _, test, _ = evaluate_test_set(ck, CLIENT_DB[ck], L)
        return rmse, run_cases(ck, L, cases), fit_history(ck), test[["actual", "pred"]]
    finally:
        ml_core.ARTIFACT_DIR = old_dir
        ml_core._model_cache.clear()
        shutil.rmtree(tmp, ignore_errors=True)


def run_settings_tests(ck):
    """
    For every Test column on the settings tab: train a temporary model with
    those settings (test-case days excluded), predict every case dish, and
    write the Results / Category accuracy tables. Production artifacts are
    never touched.
    """
    info, L = CLIENT_DB[ck], get_logic(ck)
    path, _ = ensure_workbook(ck, L)
    cases = read_frame(path, "cases")
    # Config-1 always mirrors production (model_params.json); other columns are experiments.
    tests = [("Config-1", "production — the model the app uses now; every other config must beat it",
              ml_core.get_params(ck))]
    tests += [t for t in parse_settings(path) if t[0] != "Config-1"]
    write_workbook(path, pd.read_excel(path, sheet_name="cases"), settings_frame(ck, L, tests))
    df = ml_core.prepare_dataset(ck, info["dataset"], L)   # case days already dropped
    case_days = pd.to_datetime(cases["date"], errors="coerce").dt.normalize()
    overlap = set(case_days.dropna()) & set(df["date"].dt.normalize())
    assert not overlap, f"test-case days leaked into training data: {sorted(overlap)}"
    n_case_days = case_days.nunique()

    results, cat_rows, summary = [], [], []
    for name, label, params in tests:
        # Each result is named after its config plus when it ran, e.g. "Config-1 2026-11-03 15:53".
        test_name = f"{name} {datetime.now():%Y-%m-%d %H:%M}"
        print(f"[{ck}] {name} ({label or 'no label'}): training…")
        internal_rmse, rows, _, _ = trial(ck, L, params, cases)

        config = label or name
        split = split_text(df, params, n_case_days)
        ptxt = params_text(params)
        for r in rows:
            ok = r.get("pred") is not None
            results.append({
                "Test": test_name, "Config used": config, "Train/Val/Test split": split,
                "Parameters": ptxt, "Date": r["date"], "Item": r["dish"], "Category": r["category"],
                "Predicted": round(r["pred"], 4) if ok else None,
                "Actual": round(r["actual"], 4),
                "Difference": round(r["signed_diff"], 4) if ok else None,
                "Difference %": round(r["signed_pct"], 1) if ok else None,
                "Note": (f"fallback: {r['fallback']}" if r["fallback"] else "") if ok else r["error"],
            })
        frame = pd.DataFrame(rows)
        for cat, g in frame.groupby("category", sort=True):
            p = g[g["pred"].notna()] if "pred" in g else g.iloc[0:0]
            cat_rows.append({"Test": test_name, "Category": cat, "Dishes": len(g), "Predicted": len(p),
                             "Accuracy %": round(float(g["acc"].mean()), 1)})
        p = frame[frame["pred"].notna()] if "pred" in frame else frame.iloc[0:0]
        overall = round(float(frame["acc"].mean()), 1)
        cat_rows.append({"Test": test_name, "Category": "ALL", "Dishes": len(frame), "Predicted": len(p),
                         "Accuracy %": overall})
        summary.append((test_name, config, split, ptxt, internal_rmse, overall, len(p), len(frame)))
        print(f"[{ck}] {test_name}: accuracy {overall}% within ±{WITHIN_PCT:.0f}% of {len(frame)} dishes "
              f"({len(p)} predictable) "
              f"(internal test RMSE {internal_rmse:.4f})")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(REPORT_DIR, ck, f"tests-{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    res_df, cat_df = pd.DataFrame(results), pd.DataFrame(cat_rows)
    # Per-day accuracy, and one sheet per test-case day (all configs side by side).
    res_df["_acc"] = (res_df["Difference %"].abs() <= WITHIN_PCT) * 100.0   # unpredicted (NaN) → miss
    day_df = (res_df.groupby(["Test", "Date"], sort=True)
              .agg(Dishes=("Item", "size"), Predicted=("Predicted", "count"),
                   **{"Accuracy %": ("_acc", "mean")})
              .round(1).reset_index())
    res_df = res_df.drop(columns="_acc")
    day_cols = ["Test", "Item", "Category", "Predicted", "Actual", "Difference", "Difference %", "Note"]
    xlsx = os.path.join(out_dir, "test_results.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        res_df.to_excel(xw, sheet_name="Results", index=False)
        cat_df.to_excel(xw, sheet_name="Category accuracy", index=False)
        day_df.to_excel(xw, sheet_name="Day accuracy", index=False)
        for day, g in res_df.groupby("Date", sort=True):
            g[day_cols].to_excel(xw, sheet_name=f"{day:%Y-%m-%d}", index=False)
        for ws in xw.sheets.values():
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 50)

    md = [f"# Settings tests — {info['name']} ({stamp})\n",
          f"Cases: `{os.path.relpath(path, _ROOT)}` · {n_case_days} days, {len(cases)} dishes. "
          f"Accuracy % = dishes within ±{WITHIN_PCT:.0f}% of actual / all dishes (unpredicted = miss).\n",
          md_table(["Test", "Config used", "Train/Val/Test split", "Parameters",
                    "internal test RMSE", "case accuracy %", "predictable"],
                   [[n, c, sp, pt, f4(rm), f1(acc), f"{k}/{t}"] for n, c, sp, pt, rm, acc, k, t in summary]),
          "", "## Results by day\n"]
    for day, g in res_df.groupby("Date", sort=True):
        accs = ", ".join(f"{r['Test']}: {f1(r['Accuracy %'])}%"
                         for _, r in day_df[day_df["Date"] == day].iterrows())
        md += [f"### {day:%a %d %b %Y} — accuracy {accs}\n",
               md_table(["Test", "Item", "Category", "Predicted", "Actual", "Difference",
                         "Difference %", "Note"],
                        [[r["Test"], r["Item"], r["Category"],
                          "—" if pd.isna(r["Predicted"]) else f"{r['Predicted']:.4f}", f"{r['Actual']:.4f}",
                          "—" if pd.isna(r["Difference"]) else f"{r['Difference']:+.4f}",
                          "—" if pd.isna(r["Difference %"]) else f"{r['Difference %']:+.1f}",
                          r["Note"] or ""]
                         for _, r in g.iterrows()]), ""]
    md += ["## Category accuracy\n",
           md_table(["Test", "Category", "Dishes", "Predicted", "Accuracy %"],
                    [[r["Test"], r["Category"], r["Dishes"], r["Predicted"],
                      f1(r["Accuracy %"])]
                     for r in cat_rows]), ""]
    with open(os.path.join(out_dir, "tests.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"   results: {xlsx}")
    return xlsx


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the per-pax model and write a report.")
    parser.add_argument("--client", default="all", help="Client key (e.g. 'tekion') or 'all'.")
    parser.add_argument("--train", action="store_true", help="Retrain before evaluating.")
    parser.add_argument("--tests", action="store_true",
                        help="Run every settings column of the test workbook instead of the report.")
    parser.add_argument("--import-cases", metavar="XLSX",
                        help="Replace the cases tab of the client's test workbook with this file.")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252

    if args.client == "all":
        targets = [k for k, v in CLIENT_DB.items() if v.get("has_embeddings")]
        if args.import_cases:
            parser.error("--import-cases needs a single --client")
    elif args.client in CLIENT_DB and CLIENT_DB[args.client].get("has_embeddings"):
        targets = [args.client]
    else:
        parser.error(f"unknown or multiplier-only client '{args.client}'")

    if args.import_cases:
        path = import_cases(targets[0], get_logic(targets[0]), args.import_cases)
        print(f"cases tab replaced: {path}\n"
              f"retrain so these days are excluded: python scripts/train.py --client {targets[0]} --force")
        return 0

    ok = True
    for ck in targets:
        try:
            if args.tests:
                run_settings_tests(ck)
            else:
                report_client(ck, retrain=args.train)
        except Exception as e:
            ok = False
            print(f"[fail] {ck}: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
