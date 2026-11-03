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
