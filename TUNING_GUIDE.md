# Per-pax model: tuning and improvement guide

What to change, what it costs, what it gave, and when to use it. This is the
*Model_Tuning_Guide.pdf* rewritten after tuning Clario (3–17 Nov 2026):
- Every item from the PDF is here, plus everything added on the way.
- The PDF's "Reward" was a guess. Here each item shows what it **measured** on Clario.
- Every number comes from [CHANGELOG.md](CHANGELOG.md).

**Status:**
- **KEPT**: in production for Clario.
- **REJECTED**: tested, no gain or worse; removed from the code.
- **NOT TRIED**
- **OPEN**: needs data or work.

**Tags:**
- `[DO FIRST]`: cheap and needed before anything else.
- `[HIGHEST IMPACT]`: the biggest measured wins.

## Principles (read first)

- **The kitchen's knowledge beat every model setting.** The PDF expected its biggest wins
  from headcount and season features. On Clario both lost. What won was data that says how
  the kitchen decides portions: standard portions (B1) and the kind of day (B2).
- **Measure; don't trust guesses.** Three of the PDF's top-rated changes did nothing or harm
  on Clario: shrinking the dish embedding (E1), the shortage-weighted loss (C4) and headcount
  or season features (B5, B6).
- **One random start proves nothing.** The seed alone moves Clario by about ±1.4%, about 6 of
  170 dishes. Use 3 seeds; for a close call, 10 seed groups compared in pairs.
- **Gains seen only on validation do not count,** and neither do gains seen only on the days
  you chose with. Confirm on a sealed test set and a walk-forward.
- **Every setting has a name** in `model_params.json`, so nothing depends on line numbers.
  The model reads them all from there.
- **Embedding width follows the number of distinct values, not importance.** Fixed widths:
  weekday 2, sub-category 4, category 4, kind of day 2. Don't widen one to make it "matter
  more".
- **Compute the no-model baseline per client:** each category's average.
- **Remove what fails, keep the record.** Rejected options are deleted from the code. Their
  configs stay on the workbook's settings tab, and their results stay in the CHANGELOG.

## Part 1: The workflow (in this order)

Don't start with the knobs: they gave Clario nothing. Work top-down:

1. **Clean the data** (section A). Wrong units, inconsistent targets, duplicate names. Nothing
   else matters while the target is dirty.
2. **Ask the kitchen** (B1, B2, B11). Standard portions per category and kind of day; how close
   is close enough; how bad a shortage is compared with waste; who the MG counts.
3. **Set aside the test days and fix the measures** (section I). Newest ~30 days, split in time
   order into Test 1, 2 and 3, the last sealed. Measure within 5 g, within ±10%, kitchen cost,
   and the no-model baseline.
4. **Build the structure** (B, C, D, G): category-offset target, the 5 g loss, masking,
   calibration, re-basing, kind of day, averaged models. One change at a time.
5. **Then the knobs** (E, F) with `scripts/search.py`, in this order: early stopping, dropout,
   embedding width, learning rate, layer sizes, batch. Expect noise.
6. **Verify** on the sealed Test 3 and an October-style walk-forward.
7. **Ship** (Part 3 checklist), then a **shadow trial** and **monthly retrains**.

## Part 2: Master list of changes

### Scoreboard

| Biggest measured wins on Clario | Gain |
|---|---|
| B1 Standard portions + re-basing | +20.7 dishes within 5 g (Test 1+2, 3 seeds) |
| B2 Kind of day from the menu + biryani-day portions | +15.3 within 5 g; October +12 of 135 |
| C1 Category-offset target | Error 0.0269 → 0.0074 kg (3 Nov data) |
| A1–A2 Units in kg, consistent targets | Error 0.253 → 0.0125 kg; the model beat the baseline for the first time |
| G3 Calibration | +8.4 within ±10% (Test 1+2, 3 seeds) |
| G1 Averaged models, 1 → 3 | +5.6 within 5 g (10 seed groups) |

| The PDF expected a lot, Clario measured | Result |
|---|---|
| B5–B6 Headcount and season features ("highest impact") | Lost on the test days; removed |
| E1 Shrink the dish embedding 8 → 4 ("biggest parameter lever") | 8 is better |
| C4 Shortage-weighted loss ("do first") | Each shortage removed cost 1.6–3.3 extra overs |
| C2 Huber, delta ~0.1 | Pseudo-Huber at 0.005 (5 g) won |

---

### A. Data and labels `[DO FIRST]`

**A1. Every category in kg per person** `[DO FIRST]` `[HIGHEST IMPACT]`
- Effort: Low · Clario: 70 snack rows were counted in pieces (about 2 per person) under kg
  categories. The error was 0.253 kg; after the kitchen re-weighed them, 0.0125 kg, beating the
  category baseline (0.0138) for the first time. **KEPT** (fixed in the file).
- Check: the median `ideal_pp` per category. Anything near 1–2 is pieces.

**A2. Target consistent with its inputs** `[DO FIRST]` `[HIGHEST IMPACT]`
- Effort: Low · Clario: one dataset file was only 53% consistent after hand edits. The clean file
  (every `ideal_pp` = (quantity − wastage) ÷ client MG) moved the mean from 0.0925 to 0.100.
  **KEPT**.
- Check: recompute `ideal_pp` and compare.

**A3. Impossible rows**
- Effort: Low · Clario: 4 rows with wastage > quantity (ideal_pp −0.58 to −0.008). Dropping
  them: 83.0 → 85.7 normal, 79.7 → 78.7 past-only (within ±10% of 129, mixed). **KEPT**: fix
  the rows in the dataset file, no code filter.

**A4. Same dish, different spellings** (`dish_map` in `client_logic.py`)
- Effort: Low (the kitchen confirms) · Clario:
  - 3 Nov: 651 → 584 names.
  - 11 Nov: 15 more, 977 → 962 (paneer butter and paneer mutter masala kept apart).
  - **KEPT**.
- Why: a split dish learns half as much from each name. The app's input goes through the same
  map.

**A5. Category name variants** (`category_map`)
- Effort: Low · Clario: "Healty Rice" → "Healthy Rice", "steamed rice" → white rice, and others.
  **KEPT**.

**A6. A dish under the wrong category** (`dish_category_map`)
- Effort: Low · Clario: Gujarati Kadi → Dal, Alasande Kalu Sambar → Sambar/Rasam, Bonda Soup →
  Veg Dry. **KEPT**.
- Still open: Tomato Gujju, a condiment (~30 g), is filed under Veg Gravy and always missed.

**A7. Test dishes the model can never predict**
- Effort: Low · Clario: Curd, Dessert, Raitha/Chutney, Salad and Welcome Drink were removed from
  the test cases; predictable dishes went 84/134 → 84/84. Salad uses a fixed 60 g
  (`salad_per_pax`). **KEPT**.

**A8. Know what a zero leftover means** `[DO FIRST]`
- Effort: Low · Clario: 48% of training rows and 62% of Test 3 dishes had zero leftover. There,
  eaten = cooked, the kitchen's plan, so true demand is hidden.
  - On rows with leftovers, the model beat the category average (63.2% vs 59.1%).
  - On zero-leftover rows it lost (61.9% vs 68.6%).
  - The kitchen says zero means "the quantity was right", but run-outs also show as zero.
- **OPEN**: needs the run-out headcount (B11).

**A9. Portions that changed over time**
- Check the median cooked per category per month. Clario's 2026 portions differed from 2025.
  That led to B1.

**A10. Train on targets rounded to 5 g**
- Effort: Low · Clario: within 107.7 vs 109.0; cost and real demand slightly worse.
  **REJECTED**.

### B. Kitchen knowledge and features `[HIGHEST IMPACT]`

**B1. Standard portions + re-basing** (`rebase: true`, `data/<client>_standard_portions.csv`) `[HIGHEST IMPACT]`
- Effort: Med (the kitchen fills a table) · Clario:
  - Within 5 g: Test 1+2 **123.7 vs 103.0**; sealed Test 3 57.0 vs 46.0.
  - Shortages about halved; cost about 18% lower.
  - Switching it off costs 21 dishes.
  - **KEPT**.
- What: before each retrain, every past row is scaled × today's standard ÷ that month's median
  cooked portion for its category. Old months then teach today's kitchen.
- The table: columns `category`, `day`, `grams`, `note`. A blank day = every day; a weekday or a
  kind of day overrides it. Mark anything unconfirmed as provisional. Editing the table marks the
  model out of date.

**B2. Kind of day from the menu** (`meal_day_from_menu = True` in the client's logic) `[HIGHEST IMPACT]`
- Effort: Med · Clario (single models, 3 seeds):
  - The kind of day takes one of 4 values: veg / non-veg biryani / non-veg gravy / non-veg dry.
  - With biryani-day portions (B3): Test 1+2 within 5 g 123.7 → **139.0**; October walk-forward
    90.3 → **102.3** of 135.
  - 4 values beat 3 (137.7) and beat the logged veg/non-veg column.
  - **KEPT**.
- What: training, the test cases and the app all work it out from the day's menu
  (`ml_core.day_kind`). Biryani wins on the rare day with two non-veg categories.

**B3. Standard portions per kind of day** (rows like `veg gravy, non-veg biryani, 120`)
- Effort: Low · Clario: biryani-day portions were estimated, then confirmed by the kitchen on
  13 Nov. Test sets 202 → 197 within 5 g, because September had been cooked closer to the old
  estimates. October walk-forward 100 → **104** of 135, kg short 39.5 → 27.8. **KEPT**: the
  truth comes first.

**B4. Calibration**: see G3.

**B5. Headcount feature** (`mg_band`) `[the PDF: HIGHEST IMPACT]`
- Effort: Med (code removed) · Clario: first tested with every test dish's MG wrongly set to 1.
  After the fix: 80.3 / 76.7 vs 83.0 / 79.7 (within ±10% of 129, normal / past-only).
  **REJECTED**, removed 11 Nov.

**B6. Season and day of month** (`month`, `month_part`) `[the PDF: HIGHEST IMPACT]`
- Effort: Med (code removed) · Clario: better on validation (0.01515 vs 0.01553), worse on the
  test days (84.5% vs 84.7%). **REJECTED**, removed 11 Nov.

**B7. Menu presence flags** (which categories and sub-categories are on today)
- Effort: Med · Clario, within 5 g on Test 1+2:

  | Version | Within 5 g | Result |
  |---|---|---|
  | Full | 116.0 vs 123.7 | Overfits |
  | Light | 125.7 (+2) | Fails the rule; loses 1 within ±10% |

  **REJECTED**.

**B8. Co-menu category layer** (each dish also sees the other dishes' categories)
- Effort: Med · Clario: with dish names, widths 2/4/8 scored 133.3 / 137.7 / 132.3. Categories
  only: 135.7 / 136.7 / 134.7. Baseline: 139.0. **REJECTED**: B2 already carries "a biryani is
  on".

**B9. Weight training rows by MG** (`weight_by_mg`)
- Effort: Low · Clario: no gain. **REJECTED**, removed.

**B10. Sub-category starting level**
- Effort: Med · Clario: 123.0 vs 123.7 within 5 g (cheaper, 208.0 vs 215.5, but fails +4).
  **REJECTED**.

**B11. People served, and the headcount at a run-out** `[HIGHEST IMPACT, expected]`
- Effort: High (a daily kitchen log) · **OPEN**. It's the only way to see shortages (A8) and
  real demand. Today the model assumes people served = client MG. Request:
  [KITCHEN_DATA_REQUEST.md](KITCHEN_DATA_REQUEST.md).

### C. Target and loss

**C1. Predict the gap from the category average** (`target_mode: category_offset`) `[DO FIRST]` `[HIGHEST IMPACT]`
- Effort: Low · Clario: error 0.0269 → **0.0074** kg (3 Nov data). Raw vs offset in the 11 Nov
  search: raw did not pass. **KEPT**.
- Why: categories differ up to 10× (bread 60 g, biryani 350 g). The network only learns the small
  adjustment per dish and day.

**C2. Pseudo-Huber loss, delta 0.005** (`loss: pseudo_huber`, `loss_delta: 0.005`) `[DO FIRST]`
- Effort: Low · Clario (6 Nov package, 3 seeds): error 0.0110 vs 0.0114; 15-day accuracy 86.6%
  vs 84.4%. **KEPT**.
- What: errors below 5 g are treated like MSE, above like MAE. The training aims at the kitchen's
  5 g target. The PDF suggested Huber with delta ~0.1, which is 20× looser.

**C3. Other losses** (`mse`, `mae`, `huber`; delta 0.0025–0.05)
- Effort: Low · Clario: none passed in the 11 Nov search. **REJECTED**.

**C4. Shortage-weighted loss** `[the PDF: DO FIRST]`
- Effort: Med · Clario, within ±10% of 129 (normal / past-only):

  | Weight | Shortages | Overs | Within ±10% |
  |---|---|---|---|
  | None (production) | 23.7 | 22.3 | 83.0 / 79.7 |
  | ×1.25 | — | — | Unstable: one seed had 11 short and 41 over |
  | ×1.5 | 15.0 | 36.0 | 78.0 / 69.7 |
  | ×2 | — | — | Worse |

  **REJECTED**: each shortage removed cost 1.6–3.3 extra overs. Shortages are handled by the
  kitchen cost in the keep rule (I4) instead.

**C5. Softplus output** (always above zero)
- Effort: Low · Clario: no gain; only works with the raw target. **REJECTED**, removed.
  The output stays linear: never ReLU on the output.

### D. Co-menu and padding

**D1. Ignore empty co-menu slots** (`mask_padding: true`) `[DO FIRST]`
- Effort: Low · Clario: alone, no change (84.6% vs 84.7%, 4 Nov). Kept as part of the 6 Nov
  package. Checked: with masking, the pad slots change predictions by exactly 0 (unmasked: up to
  0.24 kg).
  - **Fixed 12 Nov:** Keras' masked average gave 0 ÷ 0 = NaN on a one-dish menu. `masked_mean`
    now returns "nothing else on the menu".
  - **KEPT**.

**D2. Zero the pad embedding** (the PDF's cheaper alternative)
- **NOT TRIED**: D1 makes it unnecessary.

**D3. Co-menu length** (`CTX_LEN` = 10)
- Clario's menus have 7–10 dishes, so 10 covers a day. **KEPT**.

**D4. Pooling type: max instead of average**
- Effort: Low · Clario: 74.3 / 71.3 vs 83.0 / 79.7. **REJECTED**. Attention pooling **NOT
  TRIED**.

**D5. Share the dish and co-menu embeddings**
- Effort: Low · Clario: 81.3 / 75.3 vs 83.0 / 79.7. **REJECTED**.

**D6. Does the model read the menu?**: see I10. Clario today: no (swapped 205 vs real 200).

### E. Network size and regularisation

**E1. Dish embedding width** (`item_emb_dim`: 4 / 8 / 16) `[the PDF: DO FIRST, shrink 8 → 4]`
- Effort: Low · Clario: 4 → **8** in the 6 Nov package. A 3 Nov re-tune with 16 (plus other
  changes) was worse. 4/8/16 in the search: none beat 8. **KEPT at 8**.

**E2. Layer sizes** (`dense_units`)
- Effort: Low · Clario: 32/16 → **64/32** (6 Nov). 128/64: **+6.7 on Test 1+2, 0 on the sealed
  Test 3**, which is overfitting the tests. **KEPT at 64/32**.

**E3. Dropout** (`dropout`)
- Effort: Low · Clario: 0.5 → **0.1** (6 Nov). The staged tuning's 0.3 → 0.1 had only won on
  validation. Search 0.1–0.5: none passed. **KEPT at 0.1**.
- When: raise it if overfitting, lower it if underfitting.

**E4. Number of hidden layers**
- Effort: Med · Clario: 1 layer 76.7 / 72.0; 3 layers 78.3 / 80.3; production 2 layers
  83.0 / 79.7. **REJECTED**.

**E5. Batch normalisation**
- Effort: Med · Clario: no gain. **REJECTED**, removed.

**E6. L2 weight decay, a second dropout**
- **NOT TRIED** as such. AdamW (weight decay) was in the search: no gain.

**E7. Activation** (`relu`, `leaky_relu`, `elu`, `gelu`, `tanh`)
- Effort: Low · Clario: none passed the search. **KEPT: relu**.

### F. Training loop and optimizer

**F1. Early stopping** (`early_stopping`: patience 5, `epochs` = a maximum of 100) `[DO FIRST]`
- Effort: Low · Always on, keeps the best epoch's weights. Error before → after (kg):

  | Client | Before | After |
  |---|---|---|
  | Clario | 0.0117 | 0.0114 |
  | Tekion | 0.0102 | 0.0097 |
  | Odessia | 0.0169 | 0.0145 |
  | Tessolve | 0.0516 | 0.0230 |
  | Stripe | 0.345 | 0.323 |
  | Rippling | 0.0710 | **0.0883 (worse)** |

  Patience 3/5/8 in the search: none passed. **KEPT**.

**F2. Learning rate** (`learning_rate`)
- Effort: Low · Clario: 0.003. Search 0.0003–0.01: none passed. **KEPT**.

**F3. Batch size** (`batch`)
- Effort: Low · Clario: 32. Search 16/64: none passed. **KEPT**.

**F4. Optimizer** (`adam`, `adamw`, `rmsprop`)
- Effort: Low · Clario: none beat Adam. **KEPT: adam**.

**F5. Learning-rate schedule, ReduceLROnPlateau**
- **NOT TRIED**: early stopping ends training around epochs 8–15.

**F6. ModelCheckpoint**
- Covered: early stopping restores the best epoch's weights.

**F7. See training per epoch**
- `python scripts/train.py --verbose`; the learning curve on the Model Test page.

**F8. Fixed seed** (`seed`: 42)
- The same seed gives the same model. The train/test split always uses 42, so runs stay
  comparable. **KEPT**.

### G. After training: averaging, calibration, rounding

**G1. Averaged models** (`ensemble`)
- Effort: Low (training takes N× as long) · Clario:
  - 11 Nov: 6 single models averaged 123.5 within 5 g; averaged pairs 128.5 (+5.0), ±10% +3.2,
    kg cost −5.9%.
  - 17 Nov, 10 fresh seed groups:

    | Models | Within 5 g, Test 1+2 | Within ±10% | Kg cost |
    |---|---|---|---|
    | 1 | 131.9 | — | — |
    | 3 | 137.5 | 150.6 | — |
    | 5 | 137.7 | 152.3 | about 2% lower than 3 |

  - Live with 5 models: 200/255.
  - **KEPT at 5**: chosen over the rule's bar, reason written down.
- What: same data and settings, different seeds; the app averages their **answers**.

**G2. Average the models' weights into one model**
- Effort: Low · Clario: 149/255 vs 197 for averaging answers. **REJECTED**: each model stores
  what it learns in different hidden units, so averaging weights blurs dishes together.

**G3. Calibration** (`calibrate: true`) (the PDF: "bias correction / tilt") `[HIGHEST IMPACT]`
- Effort: Low · Clario: Test 1+2 within ±10% 117.7 vs 109.3; sealed Test 3 53.0 vs 50.7. Off
  costs 7 dishes within 5 g. **KEPT**.
- What: each category's median miss on the validation rows (the newest 10% of training days),
  shrunk by n/(n+10), is added to its predictions. It follows the current kitchen instead of a
  blend of old and new.

**G4. Round the output to 5 g** (nearest 0.005)
- Effort: Low · Clario: rounding down instead cost 0.3% accuracy and gave more shortages.
  **KEPT**: nearest, for predictions and actuals in the tests.

### H. Serving and speed

**H1. Call the model directly** (`m(x, training=False)`, not `model.predict`)
- Effort: Low · Clario: about 6× faster per dish, same numbers. The Model Test page went from
  62 s to 14 s. **KEPT**.

**H2. Predict the whole menu in one call** `[the PDF: DO FIRST]`
- Effort: Low–Med · **OPEN**: still one call per dish per model, about 0.5 s for a 7-dish day
  with 3 models (5 models about 0.8 s, scaled up, not timed). One call per model would give the
  same numbers.

**H3. Load models once, reload after a retrain**
- `_model_cache`, keyed by file time. **KEPT**.

**H4. A dish the model has never seen**
- Effort: Low · The first known dish of its sub-category is used. Its sub-category median
  instead: within ±10% 79 → 76. **REJECTED**; the first-dish rule stays.

### I. Evaluation: how to judge better or worse

**I1. Dishes within 5 g** (prediction and actual both rounded to 0.005) `[DO FIRST]`
- The primary measure since 11 Nov: one serving-spoon step, the kitchen's own target. It replaced
  RMSE and then ±10% as the lead.

**I2. Dishes within ±10%**
- Must not drop. It guards the big dishes, where 5 g is under 2%. Tightened from ±15% on
  3 Nov.

**I3. Days on target**
- Days where more than 70% of the dishes are within 5 g.

**I4. Kitchen cost** = Σ MG × (1.5 × kg short + kg over) per person `[HIGHEST IMPACT]`
- The kitchen set the shortage weight: 1.5. The cost must fall per person, per √MG and per MG,
  so no win depends on the scaling. 1.0 and 2.0 are shown for comparison.

**I5. RMSE and MAE** (on the internal test days)
- A sanity check only: the internal days mix 2025 and 2026. A change can win on RMSE and lose in
  the kitchen.

**I6. Bias** = mean(predicted − actual)
- Plus = waste, minus = shortage. Shown per set on the Model Test page; per category it is
  what calibration (G3) corrects.

**I7. By segment** (category, kind of day, day)
- Clario Test 3:
  - **By kind of day:** veg days 40/46 within 5 g; biryani days 21/38, the weak spot.
  - **By category:** Veg Dry ran 9 g high, Sambar/Rasam 3 g low.
  - **By day:** the Model Test page shows each Test 3 day and each dish.

**I8. The no-model baseline** (each category's average) `[DO FIRST]`
- Clario today: 131/255 within 5 g vs the model's 200. On 11 Nov the model was only level with
  it (165 vs 164 within ±10%). That is what started the data work.

**I9. The kitchen's own plan** (the `per_pax` column)
- Clario Test 3: kitchen 70/84 within 5 g, model 61. On zero-leftover dishes the plan scores
  "exact" by construction, so this flatters the kitchen until run-outs are logged (A8, B11).

**I10. Menu-swap check**
- Predict each day with another day's menu. If the score doesn't drop, the model isn't using the
  other dishes. Clario: 205 swapped vs 200 real.

**I11. Seeds**
- 3 seeds (42, 7, 123) for every comparison. For a close call, 10 fresh seed groups compared in
  pairs, with a 95% range: that is how the number of averaged models was settled.

**I12. Test sets: held out, in time order, the last one sealed** `[DO FIRST]`
- `test_cases/<client>_test.xlsx`, `cases` tab, `test` column. Every date on it is dropped from
  training automatically (`ml_core.holdout_dates`); never add those days to the training data.
- Test 1+2 choose; Test 3 is opened only at the end (`scripts/search.py` does this).
- A set you have looked at many times is no longer fair.

**I13. Walk-forward**
- Train up to the end of a month, predict every day of the next. Clario: trained to 30 Sep,
  16 October days.
- It also proved monthly retraining pays: trained to 30 Sep vs 3 Sep, 94 vs 88 within 5 g.
- **OPEN**: a one-off script so far, not in the repo.

**I14. The learning curve must match the measured error**
- Clario: the chart showed √loss as "RMSE": 3.7 g, when the real figure was 7.5 g. Training now
  records the real RMSE, and the Model Test page warns if the curve and the model files
  disagree. **KEPT**.

**I15. Real-world verdict: waste %, run-outs, cost, how often the kitchen overrides** `[HIGHEST IMPACT]`
- **OPEN**: needs the shadow trial and the kitchen log (B11).

## Part 3: The one rule

1. **Write the keep rule before the run.** Default (`keeps` in `scripts/search.py`), averaged
   over 3 seeds:

   | Measure | To keep a change |
   |---|---|
   | Within 5 g | At least **4 more** dishes |
   | Within ±10% | **No fewer** |
   | Kitchen cost | At least **3% lower** under every MG scaling |
   | Internal RMSE | No more than **5% worse** |

2. **One change at a time from production**, then combine only the winners.
3. **Confirm on the sealed test set and the walk-forward.** If you overrule the rule, write
   down why.
4. **Ship:**
   - **Settings:** set them in `model_params.json` with `_chosen_at` and `_why`.
   - **Retrain:** `python scripts/train.py --client <client> --force --note "why"`.
   - **Verify:** the Model Test page loads with no warning; the scores on the test sets and
     the walk-forward; real menus, including a one-dish menu and a new dish; `pytest` passes.
   - **Record:** Config-1 on the settings tab = production; a CHANGELOG row (change / not
     changed / result, including what got worse).
   - **Commit and push**, with a tag before a big change (e.g. `clario-model-frozen-2026-11-12`).
5. **Then prove it on new days:**
   - **Shadow trial, 2–4 weeks:** note the app's numbers, cook to the kitchen's plan. Replace the
     plan for a category only if the app is within 5 g on more dishes with no more run-outs.
   - **Retrain monthly.**

A model can win on every test and still lose in the kitchen. Only the shadow trial settles it.

## Where things live

| What | Where |
|---|---|
| Settings per client | `model_params.json`; defaults in `ml_core.DEFAULT_PARAMS` (a misspelled setting stops training) |
| Spellings, categories, kind of day | `client_logic.py`: `category_map`, `dish_map`, `dish_category_map`, `meal_day_from_menu` |
| Standard portions | `data/<client>_standard_portions.csv` |
| Test days and the log of configs tried | `test_cases/<client>_test.xlsx` (`cases` and `settings` tabs) |
| Train | `python scripts/train.py --client <client> [--force] [--note "..."] [--verbose]` |
| Search settings | `python scripts/search.py --client <client>` (cached in `reports/<client>/search_runs.jsonl`) |
| Full report, settings tests | `python test_model.py --client <client> [--tests]` |
| Live check | App → Model Test |
| History | [CHANGELOG.md](CHANGELOG.md), `artifacts/<client>_training_log.csv` |
| What to ask the kitchen | [KITCHEN_DATA_REQUEST.md](KITCHEN_DATA_REQUEST.md) |

## Clario in production (17 Nov 2026)

**Settings:** category-offset target · pseudo-Huber, delta 0.005 · layers 64/32 · dropout 0.1 ·
dish embedding 8 · masking · calibration · re-basing · 5 averaged models · Adam, learning rate
0.003 · batch 32 · early stopping 5 · kind of day from the menu.

**Results:**

| Set | Within 5 g | Within ±10% |
|---|---|---|
| Test sets | 200/255 | 221/255 |
| October walk-forward | 103/135 | 118/135 |
| No-model baseline | 131/255 | — |

For a new client these settings are one candidate to test, not the default: start from
`DEFAULT_PARAMS`, record that baseline, then follow Part 1.

**Still open for Clario:**
- **Shortages:** the run-out headcount (B11).
- **The menu:** the co-menu adds nothing at present (I10).
- **Waiting on the kitchen:** everyday bread (60 or 66 g) and Wednesday Healthy Rice (100 or
  120 g).
- **Speed:** H2.
- **Walk-forward:** a repo script for I13.
