# Changelog

What changed in the model, data and tests, and what did not. Newest first.

Every production retrain is also logged automatically, one row per run, in
`artifacts/<client>_training_log.csv`. Each row records the dataset, row and
day counts, the number of test-case days held out, all settings, the test
RMSE and its change from the previous run, and a `changes` column that lists
what changed and what did not. Add your own remark with:

    python scripts/train.py --client clario --force --note "why I retrained"

Evaluation history lives in `reports/<client>/runs.json`. Results from the
10-day settings tests are in `reports/<client>/tests-<time>/`.

RMSE below is the error on the dataset's internal held-out days, in kg/person
(lower is better). "10-day accuracy" is the mean of 100 − |error %| over the
Clario test-case dishes.

---

## 2026-11-06

### Clario

| Time | Change | Not changed | Result |
|---|---|---|---|
| — | **New production model.** Loss mse → **pseudo_huber, loss_delta 0.005** (5 g); layers 32/16 → **64/32**; dropout 0.5 → **0.1**; item_emb 4 → **8**; **mask_padding on**. Chosen from the grid / seed / loss-delta tests: best on all three measures averaged over seeds 42, 7, 123. | Dataset (2), 15 test days (held out: 15 days in the training log), category_offset, lr 0.003, batch 32, adam, relu, early stopping 5 (max 100), other clients | Production retrain: RMSE **0.0110** (was 0.0114), 15-day accuracy **86.6%** (was 84.4%), 73/129 within ±10% (was 72). 3-seed average: 0.0108 / 87.2% / 80. Watch: under-predicts some Healthy Rice dishes (Broken Wheat 42 g vs 100 g, Daliya Upma 40 g vs 60 g). **In production.** Roll back by restoring the previous `model_params.json` values and running `scripts/train.py --client clario --force`. |
| — | **Settings tab: added Config-8** = the previous production model, to compare and roll back. **Fixed** a duplicate `seed` row (seed became a per-config setting but was still listed as fixed), which crashed reading the settings tab back. | Cases tab, Config-1 to Config-7 | Config-1 = the new production settings. New test `tests/test_settings_workbook.py`; 75 tests pass. |

## 2026-11-03

### Clario

| Time | Change | Not changed | Result |
|---|---|---|---|
| 17:31 | **"Within tolerance" changed from ±15% to ±10%.** **Dataset → `Clario_Wastage_combined_old_new (2).xlsx`**: 3,115 rows, 6 Jan 2025 – 8 Oct 2026 (the previous file minus 5 days). **Test cases → `TEST_last10days (4).xlsx`**: those 5 days (4, 17, 31 Aug; 21 Sep; 12 Oct 2026) added, so 15 days / 129 dishes. Re-tuned → category_offset, epochs 3, lr 0.003, layers 128/64, dropout 0.5, batch 64, item_emb 16. | All other rows of the dataset; the original 84 test dishes | New tuning: RMSE 0.0121, 15-day accuracy 83.4%, 71/129 within ±10%. **The previous settings (16:54) do better on the same data**: RMSE 0.0117, 85.4%, 79/129 within ±10%, winning 12 of 15 days. Defaults: 0.0143, 82.4%, 63/129. |
| 17:43 | **Switched production back to the 16:54 settings** (category_offset, epochs 5, lr 0.003, layers 32/16, dropout 0.5, batch 32, item_emb 4) because they beat the 17:31 re-tune on every measure. Settings tab reordered so that Config-1 = production, Config-2 = defaults, Config-3 = the rejected 17:31 re-tune. | Dataset (2), test cases (4) | RMSE **0.0117** (beats the baseline), 15-day accuracy **85.4%**, 79/129 within ±10%. **In production.** |
| 16:54 | **Dataset → `Clario_Wastage_combined_old_new (1).xlsx`**: 3,160 rows, 6 Jan 2025 – 12 Oct 2026, every dish in kg (Indian Bread now ~0.06 kg instead of 2 pieces). **Test cases → `TEST_last10days (3).xlsx`**: 84 dishes, bread in kg. Re-tuned (24 trials) → category_offset, epochs 5, lr 0.003, layers 32/16, dropout 0.5, batch 32, item_emb 4. | Architecture, features, other clients | RMSE 0.0125 (now **beats** the category-mean baseline of 0.0138). 10-day accuracy **83.7%** vs 78.6% (defaults) and 81.1% (previous tuning). Not comparable with the 0.0074 below: the internal test days now include harder 2026 days. **In production.** |
| 16:35 | **Reverted** to the cleaned 2025 dataset and the 15:49 settings. | Test cases (kept `(2)`) | RMSE back to 0.0074; 10-day accuracy 76.4%. |
| 16:27 | Tried `Clario_Wastage_combined_old_new.xlsx` (first version) and re-tuned. | — | RMSE **0.253**. Rejected: 70 snack rows (vada, bonda, bajji, cutlet, kabab, dahi balla, paddu, egg) were counted in pieces (~2/person) under kg categories (mostly Veg Dry). User cleaned the weights → the 16:54 file. |
| 16:20 | Test cases → `TEST_last10days (2).xlsx`: 84 dishes (removed Curd, Dessert, Raitha/Chutney, Salad, Welcome Drink, which the model never saw); Tomato Gujju relabelled Non-Veg Gravy → Veg Gravy. | Dataset, model | 84/84 dishes now predictable (was 84/134). Tomato Gujju (0.03 kg) is still a 0% miss; it is a condiment, not a gravy. |
| 15:49 | Added **category_offset** target mode (model predicts the gap from the category's training mean). Re-tuned Clario only → category_offset, epochs 11, lr 0.0003, layers 128/64, dropout 0.2, batch 64, item_emb 4. | Dataset (cleaned 2025) | RMSE 0.0269 → **0.0074**. 10-day accuracy 76.4% (raw tuning: 78.3%). |
| 14:29 | 10-day test workbook created from `TEST_last10days.xlsx` (134 dishes, 13–29 Oct 2026). Test-case dates are now excluded from training, tuning and evaluation. | — | No overlap with training dates. |
| 14:12 | First per-client tuning (raw target) → epochs 28, lr 0.01, layers 128/64, dropout 0.5, batch 16. | — | RMSE 0.0393 → 0.0269. |
| 14:01 | Dataset → `Clario_Wastage_dataset_cleaned (1).xlsx`: spelling fixes ("Healty Rice" → "Healthy Rice"), dish names merged (651 → 584), proper dates. Same 1,525 rows. | Columns, sheet | — |

### Tekion, Odessia, Rippling, Stripe, Tessolve

| Time | Change | Not changed | Result |
|---|---|---|---|
| 15:5x | **Reverted to the original default settings** (`DEFAULT_PARAMS`); removed from `model_params.json`. To be tuned one client at a time. | Datasets | RMSE: Tekion 0.0102 · Odessia 0.0169 · Rippling 0.0710 · Stripe 0.3454 · Tessolve 0.0516 |
| 14:29 | Each dataset's last 10 days moved to `test_cases/<client>_test.xlsx` and excluded from training. | Datasets, settings | Internal test split changed, so RMSE moved slightly vs the morning. |
| 14:21 | Tuned all clients (raw target). | — | Superseded by the 15:5x revert. |

### Code and tooling

- **2026-11-04 — Readable charts and cleanup.** The learning curve now shows error in kg/person (RMSE) on a linear axis, with a ring on the lowest validation epoch; best is chosen on validation, so Clario's epoch 10 (0.0139) beats epoch 14 (0.0140) even though epoch 14's training error is lower. Bar charts show every label (stacked full width, explicit heights); the scatter has rounded ticks. Removed dead code: `_fit_chart`'s unused loss table, `_bar_chart`'s unused `highlight`/`fmt` options, and `latest_report` (replaced by `latest_run`). Pruned `reports/` to each client's latest report, latest settings test and `runs.json` (7.1 MB → 1.2 MB), and deleted Python caches. Fixed the 0.005-rounding lines in `APP_UNDERSTANDING.md`.
- **2026-11-04 — Verdict made readable**: each failed check now says what happened (for example "2 categories got worse: dal (0.0101 → 0.0123), …" instead of "failed: no category regressed"). The app lists every check ✅/❌ and the run it is compared with. A re-test of an unchanged model (identical scores) now gives **UNCHANGED** instead of REJECT.
- **2026-11-04 — 15-day test cases shown in full** in Model Test: the production model is run live on the test days; overall accuracy and within ±10%, an accuracy-by-day bar chart, then one table per day with every dish (predicted, actual, difference, accuracy, within ±10%, fallback).
- **2026-11-04 — Early stopping always on** (patience 5; `epochs` = maximum 100; `early_stopping: 0` is no longer allowed). All six clients retrained; every one now gets a "good fit" verdict and ran exactly 5 epochs past its best. RMSE before → after: Tekion 0.0102 → 0.0097, Clario 0.0117 → 0.0114, Odessia 0.0169 → 0.0145, Stripe 0.345 → 0.323, Tessolve 0.0516 → 0.0230, **Rippling 0.0710 → 0.0883 (worse: it keeps epoch 3)**. Clario's `epochs` 5 → 100 so early stopping isn't capped. Training-log "changes" for the five default clients list only `epochs 20→100`; they also switched early stopping on.
- **2026-11-04 — Clario configs deduplicated, each with its reason**: 10 → 7. Removed the epochs-8 trial and the two early-stopping variants, which early stopping makes identical to production or to Config-4. Labels now say what the config is and why it is there. 15-day accuracy: Config-1 production 84.4% = Config-3 random re-tune 84.4% > features 84.1% > masking / masking + staged 83.9% > defaults / features + dropout 0.1 79.8%.
- **2026-11-04 — One complete Model Test section with graphs** in the report's order: 1 learning curves (the epochs checked after the best are shaded) · 2 test-set metrics · 3 error % by category and by weekday (bar charts, ±10% target) · 4 predicted vs actual (scatter) · 5 test cases (each config vs production, with its reason; production accuracy by day; dish-level tables in one expander) · 6 verdict. Removed the duplicate Full report, Settings tests and loss-table expanders; `python test_model.py` still writes the full report file.
- **2026-11-04 — Model Test layout back to the GitHub (PR #1) format**: Run Model Test button → overfitting chart with its verdict bar and Retrain model button → verdict line + 5 metric boxes (now live for the current model) → Full report → Settings tests. This replaces the 5-step flow described below. Error by weekday moved into report section 3.
- **2026-11-04 — Improvement plan tooling** (baseline → diagnose → features → tune in priority order → verify on test days):
  - New features via the `extra_features` setting: `month` (season), `month_part` (early/mid/late for day of month), `mg_band` (expected headcount per 50). Each model saves its feature list (`<client>_features.pkl`) and prediction uses exactly that list. The six existing models were given their base list, so nothing changed for them.
  - `python scripts/tune.py --client X --staged` tunes one setting at a time in priority order (early stopping → dropout → embedding width → learning rate → layer sizes → batch), keeping a change only if validation RMSE drops by more than 0.5%. Candidates without early stopping are now scored at their final epoch, as they would really train.
  - `python scripts/train.py --verbose` prints training and validation loss for every epoch.
  - Live check: error by weekday added next to error by category (Details).
  - Removed the dead `EMB_DIMS["menu_items"]`: dish embedding width comes from `item_emb_dim`.
  - **Clario result.** On the validation slice, all three features together beat production (0.01515 vs 0.01553), and staged tuning then changed only dropout 0.3 → 0.1 (0.0152 → 0.0139). On the 15 test days, opened last: production 84.7%, features 84.5% (Config-8), features + staged tuning 83.5% (Config-9). The validation gains did not hold on unseen days. **Production unchanged.**
  - **Same plan with mask_padding on (Config-10).** Validation: no extra features 0.01476 vs all three 0.01482, so no features; staged tuning added only early stopping (patience 3, max 80): 0.0148 → 0.0146. Test days: 83.9%, RMSE 0.0119, 65/129 within ±10%, against production's 84.7%, 0.0117, 68/129. **Production unchanged.**
- **2026-11-04 — Early stopping** (`early_stopping` setting: patience in epochs, 0 = off, which is the default; `epochs` becomes the maximum). Stops after N epochs without a better validation loss and keeps the best epoch's weights. The diagnosis reports an early-stopped run as "stopped at epoch X, kept epoch Y" instead of overfitting (2 tests added). Clario with max 60 / patience 5 (Config-6): stopped at epoch 15 and kept epoch 10; RMSE **0.0114** (best of all configs; production 0.0117), 15-day accuracy 84.4% (production 84.7%), within ±10% on **72/129** dishes (production 68). Both differences are within the ~1.5% noise band. Production not changed.
- **2026-11-04 — One live model reporter.** Model Test steps 1–4 now evaluate the model on disk automatically on every rerun (cached until the model file changes), so a retrain shows up straight away. The Run Model Test button and the duplicate report view were removed; `python test_model.py` still writes the full report. "Best epoch N" now reads "lowest so far (last epoch)" when the curve never bottomed out, which is what "still learning" means.
- **2026-11-04 — `mask_padding` setting** (default off): the co-menu average ignores the empty pad slots. Verified: with masking, changing the pad embedding changes predictions by exactly 0 (unmasked: up to 0.24 kg). Added to the tuner's search space. Clario test on 15 days: production 84.7% (RMSE 0.0117); re-tune with masking (Config-4) 84.1% (0.0121); production + masking (Config-5) 84.6% (0.0122). **No improvement: production stays.** Found along the way: a different random start alone moves Clario by about ±1.4% accuracy and ±0.0004 RMSE, so config differences smaller than that are noise.
- **2026-11-04 — 5 g rounding switched off** (`ml_core.predict`, the `floor005` line is commented out) so tests and plans use the true regression output. The report's "RMSE as served" row was removed (now identical to RMSE). Clario's 15-day accuracy unrounded: production 84.7% (was 85.4% rounded), defaults 82.2%, the epochs-8 trial 83.7%. The model's raw output slightly over-predicts (bias +0.0007 kg), so rounding down was partly correcting it.
- **2026-11-04 — Model Test is a 5-step diagnosis flow**: 1 up to date · 2 learns properly (chart) · 3 beats the category average and is unbiased · 4 every category within ±10% (worst-first table) · 5 real test days (accuracy per day per config). Each step is one green/amber line; Run Model Test and Retrain model sit side by side; everything else is in a single Details expander.
- **Retrain button in the app** (Model Test → overfitting check bar). It runs `scripts/train.py --client <client> --force` in a separate process, one run at a time across all users (`concurrency.training_slot`), and logs the note "retrained from the app". The app still never trains inside its own process (enforced by a test). On Streamlit Cloud a model retrained this way is not saved to GitHub and is lost when the app restarts.
- **Overfitting / underfitting chart in the app.** The 🧪 Model Test section shows the current model's training vs validation loss per epoch (log scale, best epoch marked, hover values, table view) with a verdict: good fit, overfitting, underfitting, or still learning. The same diagnosis drives report section 1. Training now records `target_std`, the spread of what the model learned, so the underfitting check is correct for category_offset models; the six existing models were backfilled from their own training rows without retraining. Current verdicts: Tekion, Odessia and Tessolve good fit; Clario and Rippling still learning; Stripe overfitting (best epoch 10 of 20).
- **18:0x — one place to change settings.** Editing `model_params.json` is now the only step, followed by `python scripts/train.py`:
  - `train.py` without `--force` retrains exactly the clients whose settings, dataset file or test-case days changed since the last log entry, and writes the reason into the log note (`auto-retrain: epochs 5→8`). Up-to-date clients are skipped.
  - The app warns when a client's model is out of date, and loads the retrained model files without a restart.
  - A misspelled or invalid setting (for example `"epoch"`, `optimizer: "adm"`, or `dropout: 1.5`) stops with a clear error instead of being silently ignored.
  - In the test workbook, Config-1 is rewritten from `model_params.json` on every `--tests` run, so it always equals production.
- `test_model.py`: evaluation report (learning curves, test-set metrics vs the category-mean baseline, per-category breakdown, predicted-vs-actual chart, per-day test-case tables, verdict), plus `--tests` (one throwaway model per settings `Config-N` column) and `--import-cases`. Business metrics were removed at 16:1x.
- `scripts/tune.py`: per-client hyperparameter search → `model_params.json`.
- `ml_core.py`: per-client settings (`model_params.json`), configurable activation / loss / optimizer / splits, the `category_offset` target mode, the test-case holdout, and the training log.
- App: a 🧪 Model Test section showing the latest report and settings tests, with one tab per day.
- Datasets moved to `data/`. Fixed Windows encoding errors in `app.py` (`styles.css`) and `scripts/train.py` (console output).

### Known issues

- Tomato Gujju (0.03 kg/person, a condiment) is filed under Veg Gravy and is always mispredicted.
- Non-Clario clients still lose to the category-mean baseline (except Tekion and Rippling); they have not been tuned yet.
- The production models from before today were overwritten. Every client has been retrained since, so no model from before today remains.
