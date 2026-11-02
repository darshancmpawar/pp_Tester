"""
Planner — production plan builders supporting all client vendor MG methods.

Supports: tekion_2group, 3group, day_based, toasttab_formula, and None.
All functions receive a logic object so business rules are pluggable per client.

Glossary:
  MG  — meal group count (number of people eating that meal)
  PP  — per-pax quantity (portion size per person)
  veg_mg / nonveg_mg  — separate MG figures for veg vs non-veg tracks
  Ordered Qty = Vendor PP × Vendor MG  (what the kitchen actually prepares)
"""
import math
import pandas as pd
from ml_core import norm, cats_match


# ── Rounding helpers ──────────────────────────────────────────────────────────
# TUNING: the two granularities the whole app snaps to.
#   MG  → multiples of 5 heads. Kitchens plan in batches, not individuals, and
#         a headcount of 287 reads as false precision on an ops sheet.
#   PP  → multiples of 0.005 kg (5 g). Finer than that is below what a serving
#         spoon can distinguish, so it is noise in the plan.
# Changing either changes every plan the app prints. Keep them in step with
# ml_core.floor005(), which applies the same 5 g grid to model output.

def round_to_nearest_5(x):
    """Round x to the nearest multiple of 5 (used for MG figures)."""
    return int(round(x / 5.0) * 5)

def ceil_to_next_5(x):
    """Ceiling x up to the next multiple of 5 (ensures MG never undershoots)."""
    return int(math.ceil(x / 5.0) * 5)

def round_to_005(v):
    """Round v to the nearest 0.005 (used for Vendor PP in day_based plans)."""
    return round(v * 200) / 200.0   # ×200 == ÷0.005

def vendor_pp_from_bump(client_pp, bump):
    """
    Vendor PP = ceil((client_pp + bump) × 100) / 100.

    The bump (logic.default_bump / logic.biryani_bump) adds a safety margin
    above the model's client PP. The ceil to 0.01 kg always rounds UP — never
    to nearest — so rounding can only ever add food, never remove it.
    """
    return math.ceil((client_pp + bump) * 100) / 100.0

def avg_vendor_mg(rows, key="Vendor MG"):
    """Average Vendor MG across a list of result dicts; returns 0 for empty list."""
    return sum(r[key] for r in rows) / len(rows) if rows else 0


# ── Category classification ───────────────────────────────────────────────────

def classify_rows(results, nonveg_cat, star_cats):
    """
    Split result rows into three tracks used by vendor MG formulas:
      nonveg_rows  — rows whose category matches the non-veg category
      star_rows    — rows in the client's star (premium) categories
      rest_rows    — everything else
    """
    nonveg_rows = [r for r in results if cats_match(r["Category"], nonveg_cat)]
    star_rows   = [r for r in results if norm(r["Category"]) in star_cats]
    rest_rows   = [r for r in results if r not in nonveg_rows and r not in star_rows]
    return nonveg_rows, star_rows, rest_rows


# ── Single result row ─────────────────────────────────────────────────────────

def build_row(client_pp, total_qty, category, item, is_nonveg_day, nonveg_cat, logic):
    """
    Build one prediction result dict.

    Vendor PP = client_pp + bump (biryani gets a larger bump on non-veg days).
    Vendor MG = total_qty / vendor_pp  (back-calculated from the total).
    """
    is_biryani = (
        is_nonveg_day
        and nonveg_cat
        and norm(category) == norm(nonveg_cat)
        and norm(nonveg_cat) == "non veg biryani"
    )
    bump       = logic.biryani_bump if is_biryani else logic.default_bump
    vendor_pp  = vendor_pp_from_bump(client_pp, bump)
    # Back-calculate the headcount the vendor PP implies. vendor_pp is a ceil of
    # a positive number so it is practically never 0; the epsilon only stops a
    # ZeroDivisionError killing the whole plan if a PP of 0 ever reaches here.
    vendor_mg  = total_qty / max(vendor_pp, 1e-9)
    return {
        "Category":  category,
        "Item":      item,
        "Client PP": client_pp,
        "Vendor PP": vendor_pp,
        "Total Qty": total_qty,
        "Vendor MG": vendor_mg,
    }


# ── Client plan ───────────────────────────────────────────────────────────────

def client_plan(df):
    """Standard client plan: category, item, client PP, and total qty columns."""
    return df[["Category", "Item", "Client PP", "Total Qty"]].copy()


def fixed_pp_client_plan(df, fixed_pp_map, meal_group):
    """
    Fixed-PP client plan for clients like Odessia that use a lookup table
    instead of model-predicted PP values.
    """
    plan = df[["Category", "Item"]].copy()
    # 0.10 kg/person for a category absent from the contract table — the middle
    # of the observed PP range, so an un-contracted dish bills sanely instead of
    # at zero. A category hitting this default repeatedly belongs in the map.
    plan["Client PP"] = plan["Category"].map(
        lambda cat: fixed_pp_map.get(norm(cat), 0.10)
    )
    plan["Total Qty"] = (plan["Client PP"] * meal_group).round(1)
    return plan


# ═══════════════════════════════════════════════════════════════════════════════
# VENDOR MG COMPUTATION — per-method implementations
# ═══════════════════════════════════════════════════════════════════════════════

def _tekion_vendor_mgs(results, is_nonveg_day, nonveg_cat, logic):
    """
    Tekion 2-group formula:
      Veg MG   = (avg star MG + avg rest MG) / 2   then tier-adjusted
      Nonveg MG = avg nonveg MG                     then tier-adjusted
    On non-veg days the veg MG skips the tier adjustment to avoid double-penalising.
    """
    nonveg_rows, star_rows, rest_rows = classify_rows(results, nonveg_cat, logic.star_categories)

    raw_veg_mg    = (avg_vendor_mg(star_rows) + avg_vendor_mg(rest_rows)) / 2
    raw_nonveg_mg = avg_vendor_mg(nonveg_rows)

    # On non-veg days, pass veg MG through unadjusted — the kitchen is splitting
    # its capacity; adjusting it again would under-order.
    final_veg_mg    = raw_veg_mg if is_nonveg_day else max(logic.adjust_vendor_mg(raw_veg_mg), 0)
    final_nonveg_mg = max(logic.adjust_nonveg_vendor_mg(raw_nonveg_mg), 0)
    return final_veg_mg, final_nonveg_mg


def _3group_vendor_mg(results, nonveg_cat, logic):
    """
    Odessia / Rippling 3-group formula:
      MG = (avg nonveg MG + avg star MG + avg rest MG) / 3  then tier-adjusted
    """
    nonveg_rows, star_rows, rest_rows = classify_rows(results, nonveg_cat, logic.star_categories)
    raw_mg = (avg_vendor_mg(nonveg_rows) + avg_vendor_mg(star_rows) + avg_vendor_mg(rest_rows)) / 3
    return max(logic.adjust_vendor_mg(raw_mg), 0)


def _day_based_vendor_mg(client_mg, weekday, logic):
    """
    Stripe day-based formula:
      Vendor MG = client MG × (1 - weekday_reduction_pct)
    Each weekday has a configured reduction percentage reflecting lower/higher demand.
    """
    # 0.22 (22%) is the fallback for a weekday absent from the table — i.e.
    # Saturday/Sunday, which these sites do not normally serve. It is the
    # midpoint of the configured Mon–Fri values, so it never surprises.
    reduction_pct = logic.day_reductions.get(norm(weekday), 0.22)
    return client_mg * (1 - reduction_pct)


# ── Vendor plan builder (dispatches to the correct method) ───────────────────

def vendor_plan(df, results, client_mg, is_nonveg_day, nonveg_cat, logic, weekday=None):
    """
    Build the vendor plan DataFrame and return (plan_df, veg_mg, nonveg_mg).

    Ordered Qty = max(Vendor PP × Vendor MG, Total Qty)
    — the kitchen must always cook at least the client's guaranteed quantity.
    """
    method = logic.vendor_mg_method

    if method == "tekion_2group":
        final_veg_mg, final_nonveg_mg = _tekion_vendor_mgs(
            results, is_nonveg_day, nonveg_cat, logic
        )
        nonveg_rows, _, _ = classify_rows(results, nonveg_cat, logic.star_categories)

        veg_mg    = round_to_nearest_5(final_veg_mg)
        nonveg_mg = round_to_nearest_5(final_nonveg_mg) if (is_nonveg_day and nonveg_rows) else 0

        # Floor check: combined MG must not fall below vendor_floor_ratio × client MG.
        # If it does, lift the veg portion so that veg_mg + nonveg_mg hits the floor.
        min_combined_mg = logic.vendor_floor_ratio * client_mg
        if (veg_mg + nonveg_mg) < min_combined_mg:
            veg_mg = ceil_to_next_5(min_combined_mg - nonveg_mg)

        plan = df.copy()
        plan["Ordered Qty"] = plan.apply(
            lambda row: max(
                (nonveg_mg if (is_nonveg_day and nonveg_rows and cats_match(row["Category"], nonveg_cat))
                 else veg_mg)
                * row["Vendor PP"],
                row["Total Qty"],
            ),
            axis=1,
        ).round(1)
        plan = plan[["Category", "Item", "Vendor PP", "Ordered Qty"]]
        # Return the net veg-only allocation (veg_mg − nonveg_mg) so the chip label
        # matches the client plan convention ("Veg MG" = headcount for pure-veg eaters).
        # Ordered qty for veg rows still uses the full veg_mg internally (above).
        return plan, max(0, veg_mg - nonveg_mg), nonveg_mg

    elif method == "3group":
        final_mg = _3group_vendor_mg(results, nonveg_cat, logic)
        vendor_mg = round_to_nearest_5(final_mg)

        plan = df.copy()
        plan["Ordered Qty"] = plan.apply(
            lambda row: max(row["Vendor PP"] * vendor_mg, row["Total Qty"]), axis=1
        ).round(1)
        plan = plan[["Category", "Item", "Vendor PP", "Ordered Qty"]]
        return plan, vendor_mg, 0

    elif method == "day_based":
        raw_mg    = _day_based_vendor_mg(client_mg, weekday, logic)
        # Floor of 5 heads: with a small client MG the reduction can round down to
        # 0, and a 0 MG would make every Vendor PP division below blow up.
        vendor_mg = max(5, round_to_nearest_5(raw_mg))

        # For day_based, Vendor PP is derived from the total rather than set by bump.
        plan = df.copy()
        plan["Vendor PP"]   = plan["Total Qty"].apply(lambda tq: round_to_005(tq / vendor_mg))
        plan["Ordered Qty"] = (plan["Vendor PP"] * vendor_mg).round(1)
        # Still ensure ordered qty never falls below the client's guaranteed total.
        plan["Ordered Qty"] = plan.apply(
            lambda row: max(row["Ordered Qty"], row["Total Qty"]), axis=1
        )
        plan = plan[["Category", "Item", "Vendor PP", "Ordered Qty"]]
        return plan, vendor_mg, 0

    # Unknown method — return df unchanged with zero MGs.
    return df, 0, 0


# ── Aggressive plan builder ───────────────────────────────────────────────────

def aggressive_plan(vendor_df, results, final_veg_mg, avg_nonveg_mg,
                    is_nonveg_day, nonveg_cat, logic, method_groups=2):
    """
    Aggressive plan: bump ordered qty for star/nonveg categories, then
    back-calculate adjusted Vendor MG and Vendor PP.

    The adjusted MG figures are what the vendor commits to (higher than the
    standard vendor plan, reflecting the premium categories' extra demand).

    method_groups=2  → Tekion-style: veg and nonveg MGs computed separately
    method_groups=3  → 3-group style: single blended MG
    """
    # Which categories get a bump: star + non-veg (if applicable).
    bumped_cats = set(logic.star_categories)
    if nonveg_cat:
        bumped_cats.add(norm(nonveg_cat))

    plan = vendor_df.copy()
    plan["Ordered Qty"] = plan.apply(
        lambda row: logic.aggressive_bump(row["Ordered Qty"])
        if norm(row["Category"]) in bumped_cats
        else row["Ordered Qty"],
        axis=1,
    ).round(1)

    def _safe_div(numerator, denominator):
        return (numerator / denominator) if (denominator is not None and denominator > 0) else 0

    # AMG ("aggressive MG") = the headcount each bumped Ordered Qty implies at
    # the unchanged Vendor PP. Bumping qty without changing PP raises the
    # implied headcount; the rest of this function measures that rise and
    # hands it back as a lower MG, so the extra food shows up as PP, not heads.
    plan["AMG"] = plan.apply(lambda row: _safe_div(row["Ordered Qty"], row["Vendor PP"]), axis=1)

    # Re-classify the bumped plan rows for MG averaging.
    if nonveg_cat:
        nonveg_agg = plan[plan["Category"].notna() & plan["Category"].str.strip().str.lower().eq(norm(nonveg_cat))]
    else:
        nonveg_agg = pd.DataFrame(columns=plan.columns)
    star_agg = plan[plan["Category"].str.lower().str.strip().isin(logic.star_categories)]
    rest_agg = plan[~plan.index.isin(nonveg_agg.index) & ~plan.index.isin(star_agg.index)]

    def _mean_amg(subset):
        return subset["AMG"].mean() if not subset.empty else 0

    if method_groups == 2:
        # Blended veg MG = average of (star AMG + rest AMG) / 2.
        # The "adjustment" is how far the aggressive MG overshoots the standard MG;
        # we subtract that same overshoot from the final adjusted MG so the vendor
        # plan remains self-consistent.
        blended_veg_amg  = (_mean_amg(star_agg) + _mean_amg(rest_agg)) / 2
        nonveg_overshoot = _mean_amg(nonveg_agg) - avg_nonveg_mg
        veg_overshoot    = blended_veg_amg - final_veg_mg
        adj_combined_mg  = max(0, final_veg_mg - veg_overshoot)
        adj_nonveg_mg    = max(0, avg_nonveg_mg - nonveg_overshoot)
    else:
        # Single blended MG across all three groups.
        blended_amg     = (_mean_amg(nonveg_agg) + _mean_amg(star_agg) + _mean_amg(rest_agg)) / 3
        overshoot       = blended_amg - final_veg_mg
        adj_combined_mg = max(0, final_veg_mg - overshoot)
        adj_nonveg_mg   = 0

    # Round both to nearest 5; ensure non-veg MG never exceeds combined MG.
    adj_combined_mg = round_to_nearest_5(adj_combined_mg)
    adj_nonveg_mg   = round_to_nearest_5(adj_nonveg_mg)
    if adj_nonveg_mg > adj_combined_mg:
        adj_nonveg_mg = adj_combined_mg
    adj_veg_mg = adj_combined_mg - adj_nonveg_mg

    # Re-derive Vendor PP from the bumped ordered qty and the adjusted MG.
    def _recalc_vendor_pp(row):
        category = norm(row["Category"])
        mg_for_row = adj_nonveg_mg if (is_nonveg_day and nonveg_cat and category == norm(nonveg_cat)) \
                     else adj_combined_mg
        return round_to_005(_safe_div(row["Ordered Qty"], mg_for_row))

    plan["Vendor PP"] = plan.apply(_recalc_vendor_pp, axis=1)
    plan = plan[["Category", "Item", "Vendor PP", "Ordered Qty"]]
    return plan, adj_combined_mg, adj_nonveg_mg, adj_veg_mg


# ── Special day MG adjustment ─────────────────────────────────────────────────

def special_day_mg(client_mg, day_type, holiday_type, weekday, logic):
    """
    Return (adjusted_mg, reduction_pct).
    Looks up the reduction % from the client's special-day matrix and applies it.
    """
    reduction_pct = logic.get_reduction_pct(day_type, holiday_type, weekday)
    adjusted_mg   = client_mg * (100 - reduction_pct) / 100.0
    return adjusted_mg, reduction_pct
