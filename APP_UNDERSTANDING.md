# App Understanding

## Overview
This Streamlit app predicts **per-pax quantity (PP)** for cafeteria menu items and builds operational plans for selected clients:

- Client Production Plan
- Vendor Production Plan
- Aggressive Vendor Plan
- Special Day MG adjustment

The app is **multi-client** and client behavior is controlled by configuration + logic classes.

---

## File Responsibilities

## `app.py`
UI orchestration and runtime flow:
- client selection
- artifact check and on-demand training
- input collection (date, day metadata, menu items, non-veg options)
- predictions
- plan generation and display

## `client_database.py`
Client registry:
- display name
- dataset path
- lookup helpers

## `client_logic.py`
Per-client behavior model:
- model feature schema
- category canonicalization
- non-veg behavior
- vendor MG method
- special-day reduction policy
- optional fixed PP map

## `ml_core.py`
Model training + inference:
- dynamic embedding model based on `encoder_columns`
- encoder/map artifact generation
- fallback inference for unseen items
- client-prefixed artifact loading

## `planner.py`
Operational math layer:
- PP/qty row building
- client plan
- vendor plan across multiple MG methods
- aggressive plan
- special-day MG reduction

---

## Runtime Flow

1. User selects a client.
2. App resolves:
   - client key (`CK`)
   - client info (`INFO`)
   - client logic (`L`)
3. App validates mode consistency (`client_database` vs `client_logic`).
4. If client is multiplier-only, app runs Toasttab branch and exits.
5. Otherwise:
   - verify model artifacts exist
   - if missing, show a repair message and stop (the runtime never trains;
     artifacts come from `scripts/train.py` / the Train Models Action)
6. Load maps:
   - item → sub_category
   - item → category
   - category → sub_categories
7. Collect input:
   - date/day/month
   - day type + holiday type (if enabled)
   - non-veg options (if enabled/applicable)
   - category-wise menu items
8. Predict each item PP + total quantity.
9. Build and render plans:
   - client plan
   - vendor plan
   - aggressive plan (if enabled)
10. Optional independent Special Day button computes adjusted vendor MG.

---

## Client Mode Matrix

## Tekion
- Embedding model
- Non-veg toggle
- Vendor plan enabled
- Aggressive plan enabled
- Special day enabled
- Vendor MG method: `tekion_2group`
- Separate non-veg MG track

## Clario
- Embedding model
- Client plan only
- Vendor plan disabled
- Aggressive plan disabled
- Special day disabled

## Odessia
- Embedding model
- Non-veg section without toggle
- Vendor plan enabled
- Aggressive plan enabled
- Special day enabled
- Vendor MG method: `3group`
- Fixed PP map enabled

## Rippling
- Embedding model
- Vendor plan enabled
- Aggressive plan enabled
- Special day enabled
- Vendor MG method: `3group`
- Uses display categories for North/South Veg dry

## Stripe
- Embedding model
- Two non-veg inputs
- Vendor plan enabled
- Aggressive plan enabled
- Special day enabled
- Vendor MG method: `day_based`

## Tessolve
- Embedding model
- Client plan only
- Vendor/aggressive/special day disabled

## Toasttab
- Multiplier-only path
- No ML artifacts or prediction model

---

## Input State Model

The app uses **client-namespaced Streamlit keys**:
- prevents stale widget state leakage when switching clients
- format: `"{client_key}::{widget_name}"`

Menu entries are normalized into a single dict schema:

- `item`
- `subcat`
- `category` (canonical/internal)
- `mg`
- `display_category`
- `needs_shared_mg`

This avoids tuple-shape mismatches and simplifies downstream logic.

---

## Model Architecture

The prediction model is a dynamic, multi-input embedding regressor:

Inputs:
- target item id (`menu_items`)
- context items (`ctx`, up to 10 items)
- extra categorical features from `encoder_columns`

Embedding pattern:
- target item embedding + flatten
- context embedding + average pooling
- embedding per extra categorical feature
- concatenate all embeddings

Head:
- Dense(64, relu)
- Dropout(0.3)
- Dense(32, relu)
- Dense(1)

Output:
- predicted `ideal_pp`
- post-processed with flooring to 0.005 granularity
- `total_qty = per_pax * mg`

---

## Artifact Strategy

Artifacts are client-prefixed and stored in artifact directory:

- `<client>_per_pax_tf_model.keras`
- `<client>_<feature>_encoder.pkl`
- `<client>_item_to_subcat.pkl`
- `<client>_item_to_cat.pkl`
- `<client>_cat_to_subs.pkl`

This enables isolated lifecycle per client.

---

## Fallback and Robustness

## Unseen item handling
If item is not in menu-item encoder:
- app attempts fallback using same sub-category item from known classes
- if unavailable, returns a clear skip error

## Dataset safety
Before training:
- app checks dataset path exists
- clear UI error shown if missing

## Config safety
- app validates mode mismatch between DB and logic
- fails fast with explicit message

---

## Planning Logic Summary

## Client plan
- standard: uses predicted PP and qty
- fixed-PP mode: uses client-defined PP map and recomputed qty

## Vendor plan methods
- `tekion_2group`
  - uses star + rest grouping, optional non-veg separation, tier adjustments, floor ratio
- `3group`
  - averages non-veg + star + rest, then adjustment
- `day_based`
  - derives MG from weekday reduction

## Aggressive plan
- bumps ordered qty for effective priority categories
- recalculates derived Vendor PP with adjusted MG outputs

## Special day MG
- button-driven computation from day type + holiday type + weekday matrix

---

## Current UI Behavior Notes

- Special Day button is independent from Predict button.
- Non-veg behavior supports:
  - toggle clients
  - no-toggle clients with forced non-veg section
  - multiple non-veg rows where configured

---

## Data Expectations

Typical dataset columns:
- `date`
- `menu_items`
- `sub_category`
- `category`
- `ideal_pp`

Optional by client:
- `day_type`
- `holiday_type`
- `meal_day`
- `meal_type`

All columns are normalized to lowercase snake case in loading pipeline.

---

## Where the Tunable Numbers Live

Every operational constant is documented beside its definition. Index:

| What you want to change | File | Name |
|---|---|---|
| Portion safety margin (kg/head) | `client_logic.py` | `default_bump`, `biryani_bump` |
| How far vendor headcount is trimmed | `client_logic.py` | `adjust_vendor_mg`, `adjust_nonveg_vendor_mg` |
| Minimum vendor MG vs client MG | `client_logic.py` | `vendor_floor_ratio` |
| Aggressive-plan extra food | `client_logic.py` | `aggressive_bump` |
| Holiday attendance drops | `client_logic.py` | `_REDUCTION`, `DEFAULT_REDUCTION_PCT` |
| Stripe weekday MG drops | `client_logic.py` | `day_reductions` |
| Fixed salad portion | `client_logic.py` | `salad_per_pax` |
| Contracted billing portions | `client_logic.py` | `fixed_pp_map` |
| Model size / training length | `ml_core.py` | `EPOCHS`, `BATCH`, `EMB_DIMS`, `CTX_LEN` |
| Output rounding grids (5 heads / 5 g) | `planner.py`, `ml_core.py` | `round_to_nearest_5`, `floor005` |
| Concurrent prediction limit | `concurrency.py` | `MAX_CONCURRENT_PREDICTS` |

Changing anything in `encoder_columns` or `EMB_DIMS` requires a retrain:
`python scripts/train.py --client all --force`.

---

## Recommended Next Production Upgrades

1. Move training to offline jobs and deploy versioned models.
2. Add date-group evaluation reports by slice (weekday/day-type/client).
3. Add schema contract tests per client.
4. Log prediction inputs/outputs for monitoring and retraining.
5. Add confidence scoring for fallback predictions.
6. Add unit tests for planner edge cases and MG safety boundaries.
