"""
UI — Streamlit app that turns a day's menu into production plans.

Reading order (top to bottom matches the on-screen order):
  1. page setup + CSS
  2. client selector      → CLIENT_KEY / CLIENT_INFO / LOGIC
  3. _render_production_plan_tab()
       a. Toasttab short-circuit (multiplier-only client, no ML)
       b. artifact check
       c. inputs:  date/context → non-veg items → shared MG → menu items
       d. Predict → per-item prediction → client / vendor / aggressive plans
       e. Special Day button (independent of Predict)

Three objects drive everything and are read-only after the selector:
  CLIENT_KEY   lowercase id, e.g. "tekion" — prefixes artifacts and widget keys
  CLIENT_INFO  CLIENT_DB row: display name, dataset path, has_embeddings
  LOGIC        the client's logic object — every tunable lives there
"""
import math
import os
import warnings
import logging

import altair as alt
import pandas as pd
import streamlit as st

from client_database import CLIENT_LIST, name_to_key, get_info
from client_logic import get_logic
from ml_core import (
    ARTIFACT_DIR, artifacts_exist, load_map, norm, fmt_cols, stale_reasons,
    cats_match, get_nv_cats, predict, test_workbook_path, day_kind, get_params, standards_path,
)
from planner import (
    build_row, client_plan, fixed_pp_client_plan, vendor_plan,
    aggressive_plan, special_day_mg, avg_vendor_mg, classify_rows, round_to_nearest_5,
)
from concurrency import predict_gate, training_slot, GateBusy
import test_model

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# ═══════════════ PAGE CONFIG — must be the first Streamlit call ═══════════════
st.set_page_config(
    page_title="Production Plan",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═══════════════ CSS ═══════════════
with open(os.path.join(os.path.dirname(__file__), "styles.css"), encoding="utf-8") as _css_file:
    st.markdown(f"<style>{_css_file.read()}</style>", unsafe_allow_html=True)


def _tekion_final_veg_mg(results, is_nonveg_day, nonveg_cat, logic):
    """
    Veg MG *before* the round-to-5 that vendor_plan() applies.

    The aggressive plan needs the unrounded figure as its baseline, otherwise
    the overshoot it subtracts is measured against a different number than the
    one the vendor plan was built from. Mirrors planner._tekion_vendor_mgs().
    """
    _, star_rows, rest_rows = classify_rows(results, nonveg_cat, logic.star_categories)
    raw_veg_mg = (avg_vendor_mg(star_rows) + avg_vendor_mg(rest_rows)) / 2
    # On a non-veg day the kitchen splits capacity; tier-adjusting again under-orders.
    return raw_veg_mg if is_nonveg_day else max(logic.adjust_vendor_mg(raw_veg_mg), 0)


# ═══════════════ HEADER ═══════════════
st.markdown("""
<div class="app-header">
    <div class="app-header-icon">🍽️</div>
    <div class="app-header-text">
        <div class="app-title">Production Plan</div>
        <div class="app-sub">Cafeteria AI &nbsp;·&nbsp; Real-time menu intelligence &amp; operational planning</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════ CLIENT SELECTOR ═══════════════
selector_col, meta_col = st.columns([2, 3])
with selector_col:
    selected_client_name = st.selectbox("Select Client", CLIENT_LIST, key="client_sel")

CLIENT_KEY  = name_to_key(selected_client_name)   # e.g. "tekion"
CLIENT_INFO = get_info(CLIENT_KEY)                # dataset path, capabilities
LOGIC       = get_logic(CLIENT_KEY)               # all per-client tunables

with meta_col:
    if LOGIC.has_embeddings:
        badge_cls, badge_label = "badge-blue", "Embedding Model"
    else:
        badge_cls, badge_label = "badge-amber", "Multiplier Only"
    st.markdown(f"""
    <div class="client-meta">
        <span class="badge {badge_cls}">{badge_label}</span>
        <span class="client-name">Active: <strong>{selected_client_name}</strong></span>
    </div>
    """, unsafe_allow_html=True)


# Config guard: client_database and client_logic must agree on whether this
# client has an ML model. Disagreeing means one of the two files was edited
# without the other — fail loudly rather than silently taking the wrong branch.
db_has_embeddings = CLIENT_INFO.get("has_embeddings", None)
if db_has_embeddings is not None and bool(db_has_embeddings) != bool(LOGIC.has_embeddings):
    st.error(
        f"Config mismatch for {selected_client_name}: "
        f"client_database.has_embeddings={db_has_embeddings} vs "
        f"client_logic.has_embeddings={LOGIC.has_embeddings}. Please align configs."
    )
    st.stop()


def key(name: str) -> str:
    """
    Namespace a Streamlit widget key to the active client.

    Streamlit keys are global to the session, so without the prefix the
    "Flavoured Rice" box would keep Tekion's text after switching to Stripe.
    """
    return f"{CLIENT_KEY}::{name}"


def cat_key(category: str) -> str:
    """Turn a category name into a widget-key-safe slug ('Sambar/Rasam' → 'sambar_rasam')."""
    return norm(str(category)).replace(" ", "_").replace("/", "_").replace("-", "_")


# ═══════════════════════════════════════════════════════
#  GENERATE PRODUCTION PLAN
# ═══════════════════════════════════════════════════════
def _render_production_plan_tab():

    # ── TOASTTAB (multiplier-only: no model, no menu, MG arithmetic only) ──
    if not LOGIC.has_embeddings:
        from datetime import date as _date, datetime
        try:
            # Kitchens are in IST; the server may not be, so pin the default date.
            from zoneinfo import ZoneInfo
            today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        except Exception:
            today = _date.today()

        st.markdown('<div class="sec"><span class="sec-icon">📅</span><span class="sec-label">Date &amp; MG Input</span></div>', unsafe_allow_html=True)
        date_col, mg_col, round_col = st.columns([1, 1, 2])
        with date_col:
            service_date = st.date_input("Date", value=today, key=key("toast_date"))
        with mg_col:
            client_mg = st.number_input(
                "Client MG", min_value=1, step=1,
                value=LOGIC.default_mg, key=key("toast_cmg"),
            )
        with round_col:
            st.markdown("<br>", unsafe_allow_html=True)
            round_to_5 = st.checkbox("Round to nearest 5", value=True, key=key("toast_r5"))

        adjusted_mg = LOGIC.toasttab_adjust(client_mg)
        vendor_mg   = round_to_nearest_5(adjusted_mg) if round_to_5 else int(adjusted_mg)

        st.markdown(
            f'<div class="ctx">📆 &nbsp;<strong>{service_date.strftime("%A, %d %b %Y")}</strong> &nbsp;·&nbsp; Vendor MG result</div>',
            unsafe_allow_html=True,
        )
        col_client, col_adjusted, col_vendor = st.columns(3)
        with col_client:    st.metric("Client MG", f"{client_mg}")
        with col_adjusted:  st.metric("Adjusted",  f"{adjusted_mg:.0f}")
        with col_vendor:    st.metric("Vendor MG", f"{vendor_mg}")
        st.stop()

    # ── ENSURE MODEL ARTIFACTS ──────────────────────────────────────────
    # The app never trains in its own process. Artifacts come from
    # `scripts/train.py` (run locally or via the Train Models GitHub Action).
    # If they're missing here, fail fast with a clear repair path.
    def _ensure_artifacts_present():
        if artifacts_exist(CLIENT_KEY, LOGIC.encoder_columns):
            return
        st.error(
            f"Model artifacts for **{selected_client_name}** are missing.\n\n"
            "They are built offline by the **Train Models** GitHub Actions "
            "workflow. Trigger it via **Actions → Train Models → Run "
            f"workflow** (client: `{CLIENT_KEY}`), wait for the auto-commit, then "
            "refresh this page."
        )
        with st.expander("Expected artifact files"):
            expected_files = [
                f"artifacts/{CLIENT_KEY}_per_pax_tf_model.keras",
                f"artifacts/{CLIENT_KEY}_item_to_subcat.pkl",
                f"artifacts/{CLIENT_KEY}_item_to_cat.pkl",
                f"artifacts/{CLIENT_KEY}_cat_to_subs.pkl",
            ] + [f"artifacts/{CLIENT_KEY}_{col}_encoder.pkl" for col in LOGIC.encoder_columns]
            st.code("\n".join(expected_files))
        st.stop()

    _ensure_artifacts_present()

    # ── STALE-MODEL WARNING ─────────────────────────────────────────────
    # model_params.json, the dataset and the test-case days are the inputs;
    # if any changed since the last training, predictions still come from
    # the old model until `python scripts/train.py` is run.
    try:
        stale = stale_reasons(CLIENT_KEY, CLIENT_INFO["dataset"])
    except ValueError as bad_settings:
        st.error(f"model_params.json is invalid — {bad_settings}")
        st.stop()
    if stale:
        st.warning(
            f"**{selected_client_name}'s model is out of date:** {'; '.join(stale)}. "
            "Predictions below still use the previous model. Run `python scripts/train.py` "
            "to retrain — the app picks up the new model automatically."
        )

    # ── LOAD MAPPINGS ───────────────────────────────────────────────────
    # Built at training time; let the UI resolve sub-category from an item name
    # and offer sub-category choices for an item the model has never seen.
    item_to_subcat      = load_map(CLIENT_KEY, "item_to_subcat")   # "veg pulao" → "pulao"
    category_to_subcats = load_map(CLIENT_KEY, "cat_to_subs")      # "veg dry"   → ["poriyal", ...]
    subcat_by_lower_item = {norm(item): sub for item, sub in item_to_subcat.items()}

    ui_categories  = list(LOGIC.fixed_categories)
    star_categories = {norm(c) for c in LOGIC.star_categories}

    # Three non-veg postures:
    #   "Required"   — always non-veg (Stripe); no toggle shown
    #   "Optional"   — user decides via toggle (Tekion)
    #   "Not Needed" — veg-only client (Clario, Tessolve)
    nonveg_mode = getattr(
        LOGIC, "custom_nonveg_mode",
        "Optional" if LOGIC.has_nonveg_toggle else "Not Needed",
    )

    # ── DATE & CONTEXT ──────────────────────────────────────────────────
    st.markdown('<div class="sec"><span class="sec-icon">📅</span><span class="sec-label">Date &amp; Context</span></div>', unsafe_allow_html=True)

    if LOGIC.has_special_day:
        date_col, day_type_col, holiday_type_col = st.columns(3)
    else:
        date_col, _spacer_col = st.columns([1, 2])

    with date_col:
        service_date = st.date_input("Service Date", key=key("sel_date"))
    weekday_name = service_date.strftime("%A")   # "Monday" — a model feature
    weekday      = norm(weekday_name)            # "monday" — encoder token

    if LOGIC.has_special_day:
        with day_type_col:
            selected_day_type = st.selectbox(
                "Day Type",
                ["Regular", "Previous Day of Holiday", "Next Day of Holiday", "Holiday"],
                key=key("day_type"),
            )
        with holiday_type_col:
            selected_holiday_type = st.selectbox(
                "Holiday Type",
                ["Not Applicable", "Non-Important Holiday", "Compulsory Holiday", "Important Holiday"],
                key=key("holiday_type"),
            )
    else:
        selected_day_type     = "Regular"
        selected_holiday_type = "Not Applicable"

    # UI labels are title-case; the encoders were fitted on canonical lowercase.
    canon_day_type     = LOGIC.canonicalize_day_type(selected_day_type)
    canon_holiday_type = LOGIC.canonicalize_holiday_type(selected_holiday_type)

    # context pill
    context_parts = [f"<strong>{weekday_name}</strong>, {service_date.strftime('%d %b %Y')}"]
    if LOGIC.has_special_day and selected_day_type != "Regular":
        context_parts.append(f"<strong>{selected_day_type}</strong> &nbsp;·&nbsp; {selected_holiday_type}")
    st.markdown(
        f'<div class="ctx">📆 &nbsp;{"&emsp;|&emsp;".join(context_parts)}</div>',
        unsafe_allow_html=True,
    )

    # ── NON-VEG STATE ───────────────────────────────────────────────────
    is_nonveg_day       = False
    meal_day            = "veg"       # model feature: what kind of day this is
    first_nonveg_item   = None
    first_nonveg_cat    = None        # drives biryani bump + vendor MG split
    nonveg_rows         = []
    nonveg_mg_total     = 0           # headcount eating non-veg (toggle clients)

    if nonveg_mode == "Required":
        is_nonveg_day = True
        meal_day      = "nonveg"
    elif nonveg_mode == "Optional" and LOGIC.has_nonveg_toggle:
        is_nonveg_day = st.toggle("🍗 Non-Veg Day?", value=True, key=key("is_nv"))
        meal_day      = "nonveg" if is_nonveg_day else "veg"

    # entries: one dict per dish to predict (non-veg first, then menu categories).
    # menu: every item name on the day — feeds each prediction's co-menu context.
    entries = []
    menu    = []

    show_nonveg_section = nonveg_mode != "Not Needed" and (
        (LOGIC.has_nonveg_toggle and is_nonveg_day)
        # No-toggle clients (Odessia/Rippling/Stripe) always show the section,
        # but only where a vendor plan exists to consume the non-veg split.
        or (not LOGIC.has_nonveg_toggle and LOGIC.nonveg_item_count > 0 and LOGIC.has_vendor_plan)
    )

    # ── NON-VEG ITEMS ───────────────────────────────────────────────────
    if show_nonveg_section:
        # Prefer categories actually marked non-veg in the dataset; fall back to
        # the hard-coded list for clients whose data has no meal_type column.
        nonveg_cat_options = (
            get_nv_cats(CLIENT_KEY, CLIENT_INFO["dataset"], LOGIC)
            if CLIENT_INFO.get("dataset") else []
        )
        if not nonveg_cat_options:
            nonveg_cat_options = list(getattr(LOGIC, "nonveg_cats", []))

        if nonveg_cat_options:
            st.markdown('<div class="sec"><span class="sec-icon">🍗</span><span class="sec-label">Non-Veg Items</span></div>', unsafe_allow_html=True)
            for nv_index in range(LOGIC.nonveg_item_count):
                cat_col, item_col = st.columns(2)
                with cat_col:
                    raw_nonveg_cat = st.selectbox(
                        f"Non-Veg Category{f' #{nv_index + 1}' if LOGIC.nonveg_item_count > 1 else ''}",
                        nonveg_cat_options,
                        key=key(f"nvc_{nv_index}"),
                    )
                with item_col:
                    nonveg_item = st.text_input(f"Non-Veg Item  ({raw_nonveg_cat}):", key=key(f"nvi_{nv_index}"))

                if not nonveg_item:
                    continue

                canon_nonveg_cat = LOGIC.canonicalize_category(raw_nonveg_cat)

                # Known item → sub-category is already on file; unknown → ask.
                if norm(nonveg_item) in subcat_by_lower_item:
                    subcat = subcat_by_lower_item[norm(nonveg_item)]
                    st.caption(f"✓ Sub-category: {subcat}")
                else:
                    subcat_options = category_to_subcats.get(canon_nonveg_cat, [])
                    subcat = (
                        st.selectbox(f"Sub-cat for '{nonveg_item}':", subcat_options, key=key(f"nvsc_{nv_index}"))
                        if subcat_options
                        else st.text_input(f"Sub-cat for '{nonveg_item}':", key=key(f"nvsc_{nv_index}_t"))
                    )

                if LOGIC.has_nonveg_toggle:
                    # Tekion-style: non-veg has its own headcount, entered per item.
                    item_mg         = st.number_input(
                        f"Client MG for '{nonveg_item}':",
                        min_value=1, step=1, value=LOGIC.default_mg, key=key(f"nvmg_{nv_index}"),
                    )
                    needs_shared_mg = False
                else:
                    # Everyone eats from the same headcount — filled in below once
                    # the shared MG widget has been rendered.
                    item_mg         = None
                    needs_shared_mg = True

                entries.append({
                    "item": nonveg_item, "subcat": subcat, "category": canon_nonveg_cat,
                    "mg": item_mg, "display_category": canon_nonveg_cat,
                    "needs_shared_mg": needs_shared_mg,
                })
                menu.append(nonveg_item)
                nonveg_rows.append({"item": nonveg_item, "cat": canon_nonveg_cat, "mg": (item_mg or 0)})
                nonveg_mg_total += (item_mg or 0)

    if nonveg_rows:
        # Plans key off a single non-veg category (the biryani bump and the
        # veg/non-veg MG split are both one-category rules), so row 0 wins.
        first_nonveg_item = nonveg_rows[0]["item"]
        first_nonveg_cat  = nonveg_rows[0]["cat"]
        if not LOGIC.has_nonveg_toggle:
            is_nonveg_day = True
            meal_day      = "nonveg"

    # ── SHARED CLIENT MG ────────────────────────────────────────────────
    st.markdown('<div class="sec"><span class="sec-icon">👥</span><span class="sec-label">Meal Group Size</span></div>', unsafe_allow_html=True)
    mg_col, _spacer_col = st.columns([1, 3])
    with mg_col:
        client_mg = st.number_input(
            "Shared Client MG",
            min_value=1, step=1, value=LOGIC.default_mg, key=key("shared_cmg"),
        )

    # Non-veg entries parked above with mg=None take the shared headcount.
    for entry in entries:
        if entry["needs_shared_mg"] or entry["mg"] is None:
            entry["mg"] = client_mg

    # ── MENU ITEMS ──────────────────────────────────────────────────────
    st.markdown('<div class="sec"><span class="sec-icon">🍱</span><span class="sec-label">Menu Items</span></div>', unsafe_allow_html=True)
    # ⭐ marks the premium categories that drive vendor MG and get the
    # aggressive bump. Fallback set is for a client that configured none.
    star_categories_ui = star_categories or {"flavour rice", "flavoured rice", "veg curry", "veg gravy"}
    north_veg_dry_prediction = None   # see the South-Veg-dry rule in the predict loop

    for ui_category in ui_categories:
        label = f"{'⭐ ' if norm(ui_category) in star_categories_ui else ''}{ui_category}"
        item  = st.text_input(label, key=key(f"item_{cat_key(ui_category)}"))
        if not item:
            continue

        menu.append(item)
        # Rippling shows "North/South Veg dry" but the model only knows "veg dry".
        model_category = LOGIC.category_display_map.get(ui_category, ui_category)
        canon_category = LOGIC.canonicalize_category(model_category)

        if canon_category == "salad":
            # Salad is never predicted (fixed PP), so its sub-category is cosmetic.
            subcat = "salad"
        elif norm(item) in subcat_by_lower_item:
            subcat = subcat_by_lower_item[norm(item)]
            st.caption(f"✓ Sub-category: {subcat}")
        else:
            subcat_options = category_to_subcats.get(canon_category, [])
            subcat = (
                st.selectbox(f"Sub-cat for '{item}' ({ui_category}):", subcat_options, key=key(f"sc_{cat_key(ui_category)}"))
                if subcat_options
                else st.text_input(f"Sub-cat for '{item}':", key=key(f"sc_{cat_key(ui_category)}_t"))
            )

        entries.append({
            "item": item, "subcat": subcat, "category": canon_category,
            "mg": client_mg, "display_category": ui_category, "needs_shared_mg": False,
        })

    # ── PREDICT BUTTON ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    button_col, _spacer_col = st.columns([1, 4])
    with button_col:
        predict_clicked = st.button("Predict", key=key("predict_btn"), use_container_width=True)

    # ── RESULTS ─────────────────────────────────────────────────────────
    if predict_clicked:
        st.markdown("<br>", unsafe_allow_html=True)
        date_label  = service_date.strftime("%a %d %b %Y")
        day_type_label = (
            f" &nbsp;·&nbsp; {selected_day_type}"
            if (LOGIC.has_special_day and selected_day_type != "Regular") else ""
        )
        st.markdown(
            f'<div class="ctx">📋 &nbsp;<strong>{selected_client_name}</strong> &nbsp;·&nbsp; {date_label}{day_type_label}</div>',
            unsafe_allow_html=True,
        )

        if not entries:
            st.warning("Add at least one menu item.")
            st.stop()

        # predict_gate caps simultaneous predict loops process-wide so the
        # 1-CPU / 1-GB Streamlit Cloud tier doesn't OOM under concurrent users.
        # Excess sessions queue here briefly; on timeout they get a clean retry msg.
        # We hold the gate ONLY across the predict loop, not the plan-render
        # below, so non-ML rendering doesn't keep slots occupied.
        results = []
        if getattr(LOGIC, "meal_day_from_menu", False):   # the model's day type: veg or the non-veg category picked
            meal_day = day_kind([e["category"] for e in entries])
        try:
            with predict_gate():
                with st.spinner("Running predictions…"):
                    for entry in entries:
                        item            = entry["item"]
                        subcat          = entry["subcat"]
                        canon_category  = entry["category"]
                        item_mg         = entry["mg"]
                        display_category = entry["display_category"]

                        if display_category == "South Veg dry" and north_veg_dry_prediction is not None:
                            # Rippling serves one veg dry recipe at two counters, so
                            # South mirrors North's PP instead of being predicted twice.
                            # Relies on "North Veg dry" preceding "South Veg dry" in
                            # RipplingLogic.fixed_categories.
                            per_pax, total_qty = north_veg_dry_prediction
                        elif norm(canon_category) == "salad":
                            # Salad PP is a fixed serving size, not a model output.
                            per_pax   = LOGIC.salad_per_pax
                            total_qty = per_pax * item_mg
                        else:
                            prediction = predict(
                                CLIENT_KEY, item, menu, item_mg, subcat, canon_category, weekday, LOGIC,
                                day_type=canon_day_type, holiday_type=canon_holiday_type, meal_day=meal_day,
                                service_date=service_date,
                            )
                            if not prediction.ok:
                                st.error(f"❌ {prediction.error}  Skipping '{item}'.")
                                continue
                            if prediction.fallback:
                                st.warning(f"⚠️ '{item}' unseen — fallback via '{prediction.fallback_item}'.")
                            per_pax   = prediction.per_pax
                            total_qty = prediction.total_qty

                        if display_category == "North Veg dry":
                            north_veg_dry_prediction = (per_pax, total_qty)

                        row = build_row(
                            per_pax, total_qty,
                            LOGIC.canonicalize_category(display_category)
                            if display_category != canon_category else canon_category,
                            item, is_nonveg_day, first_nonveg_cat, LOGIC,
                        )
                        if display_category in ("North Veg dry", "South Veg dry"):
                            # Keep the operator-facing counter name in the table.
                            row["Category"] = display_category
                        results.append(row)
        except GateBusy as gate_busy:
            st.warning(str(gate_busy))
            st.stop()

        if not results:
            st.stop()

        results_df = pd.DataFrame(results)
        results_df["Total Qty"] = results_df["Total Qty"].round(1)
        results_df["Vendor MG"] = results_df["Vendor MG"].round(0)

        # ── CLIENT PLAN ─────────────────────────────────────────────────
        if LOGIC.fixed_pp_map:
            # Odessia bills from a contracted PP table, not from the model.
            client_plan_df = fixed_pp_client_plan(results_df, LOGIC.fixed_pp_map, client_mg)
        else:
            client_plan_df = client_plan(results_df)

        if LOGIC.has_nonveg_toggle and is_nonveg_day and first_nonveg_item:
            # Toggle clients split one headcount: non-veg eaters are not veg eaters.
            veg_mg = max(client_mg - nonveg_mg_total, 0)
            mg_chips = (
                f'<span class="chip chip-blue">Veg MG: {veg_mg:.0f}</span>'
                f'<span class="chip chip-purple">Non-Veg MG: {nonveg_mg_total:.0f}</span>'
            )
        else:
            mg_chips = f'<span class="chip chip-blue">Client MG: {client_mg:.0f}</span>'

        st.markdown(
            f'<div class="plan-card"><span class="plan-label">📋 Production Plan</span>'
            f'<div class="plan-chips">{mg_chips}</div></div>',
            unsafe_allow_html=True,
        )
        # Display-only rename; internal column names stay stable for the planner.
        client_plan_display = client_plan_df.rename(
            columns={"Client PP": "Per Pax Qty(g)", "Total Qty": "Total Qty(kg)"}
        )
        st.dataframe(
            fmt_cols(client_plan_display, ["Per Pax Qty(g)", "Total Qty(kg)"]),
            use_container_width=True, hide_index=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ── VENDOR PLAN ─────────────────────────────────────────────────
        if LOGIC.has_vendor_plan:
            vendor_plan_df, vendor_veg_mg, vendor_nonveg_mg = vendor_plan(
                results_df, results, client_mg, is_nonveg_day, first_nonveg_cat, LOGIC,
                weekday=weekday,
            )

            if LOGIC.has_separate_nonveg_mg and vendor_nonveg_mg > 0:
                vendor_mg_chips = (
                    f'<span class="chip chip-blue">Veg MG: {vendor_veg_mg}</span>'
                    f'<span class="chip chip-purple">Non-Veg MG: {vendor_nonveg_mg}</span>'
                )
            elif LOGIC.vendor_mg_method == "day_based":
                vendor_mg_chips = f'<span class="chip chip-amber">Day-Based MG: {vendor_veg_mg}</span>'
            else:
                vendor_mg_chips = f'<span class="chip chip-blue">Vendor MG: {vendor_veg_mg}</span>'

            st.markdown(
                f'<div class="plan-card plan-card-vendor"><span class="plan-label">🤝 Vendor Production Plan</span>'
                f'<div class="plan-chips">{vendor_mg_chips}</div></div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                fmt_cols(vendor_plan_df, ["Vendor PP", "Ordered Qty"]),
                use_container_width=True, hide_index=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # ── AGGRESSIVE PLAN ──────────────────────────────────────────
            if LOGIC.has_aggressive_plan:
                if LOGIC.vendor_mg_method == "tekion_2group":
                    # 2-group clients need the pre-rounding veg MG and the raw
                    # non-veg average as the baselines the overshoot is measured from.
                    baseline_veg_mg    = _tekion_final_veg_mg(results, is_nonveg_day, first_nonveg_cat, LOGIC)
                    baseline_nonveg_mg = avg_vendor_mg(
                        [r for r in results if cats_match(r["Category"], first_nonveg_cat)]
                    )
                    (aggressive_df, aggressive_combined_mg,
                     aggressive_nonveg_mg, aggressive_veg_mg) = aggressive_plan(
                        vendor_plan_df, results, baseline_veg_mg, baseline_nonveg_mg,
                        is_nonveg_day, first_nonveg_cat, LOGIC, method_groups=2,
                    )
                else:
                    # 3-group / day-based clients carry a single blended MG.
                    (aggressive_df, aggressive_combined_mg,
                     aggressive_nonveg_mg, aggressive_veg_mg) = aggressive_plan(
                        vendor_plan_df, results, vendor_veg_mg, 0,
                        is_nonveg_day, first_nonveg_cat, LOGIC, method_groups=3,
                    )

                if LOGIC.has_separate_nonveg_mg and aggressive_nonveg_mg > 0:
                    aggressive_chips = (
                        f'<span class="chip chip-blue">Veg MG: {aggressive_veg_mg}</span>'
                        f'<span class="chip chip-purple">Non-Veg MG: {aggressive_nonveg_mg}</span>'
                    )
                else:
                    aggressive_chips = f'<span class="chip chip-amber">Aggressive MG: {aggressive_combined_mg}</span>'

                st.markdown(
                    f'<div class="plan-card plan-card-aggressive"><span class="plan-label">🚀 Aggressive Vendor Plan</span>'
                    f'<div class="plan-chips">{aggressive_chips}</div></div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    fmt_cols(aggressive_df, ["Vendor PP", "Ordered Qty"]),
                    use_container_width=True, hide_index=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)

    # ── SPECIAL DAY (independent button) ────────────────────────────────
    # Deliberately outside the Predict block: ops want the reduced MG figure
    # without re-running predictions.
    if LOGIC.has_special_day and selected_day_type != "Regular":
        st.divider()
        special_day_col, _spacer_col = st.columns([1, 4])
        with special_day_col:
            apply_special_day = st.button(
                "Apply Special Day Logic", key=key("apply_special_day"), use_container_width=True,
            )
        if apply_special_day:
            adjusted_mg, reduction_pct = special_day_mg(
                client_mg, selected_day_type, selected_holiday_type, weekday_name, LOGIC,
            )
            reduction_col, adjusted_col = st.columns(2)
            with reduction_col:  st.metric("Reduction",          f"{reduction_pct}%")
            with adjusted_col:   st.metric("Adjusted Vendor MG", f"{adjusted_mg:.0f}")


_render_production_plan_tab()


# ═══════════════════════════════════════════════════════
#  MODEL TEST REPORT (test_model.py)
# ═══════════════════════════════════════════════════════
def _retrain_from_ui():
    """
    Run scripts/train.py for this client in a separate process, so a training
    crash or TensorFlow memory spike cannot take the app process down with it.
    The app picks the new model files up by their timestamps (ml_core cache).
    """
    import subprocess, sys
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "scripts", "train.py"),
           "--client", CLIENT_KEY, "--force", "--note", "retrained from the app"]
    try:
        with training_slot():
            with st.spinner(f"Training {selected_client_name}… (usually under a minute)"):
                done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                      env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    except GateBusy as busy:
        st.warning(str(busy))
        return
    result = [line for line in done.stdout.splitlines() if line.startswith("[")]
    if done.returncode == 0:
        st.success(f"Retrained. {' '.join(result)}")
    else:
        st.error(f"Training failed. {' '.join(result) or done.stderr[-500:]}")


def _fit_chart(hist):
    """Training vs validation error per epoch (RMSE, g/person); the epoch early stopping kept is circled."""
    n = len(hist["val_loss"])
    if "val_rmse" in hist:   # recorded every epoch since 2026-11-12
        curves = {"Training error": hist["rmse"], "Validation error": hist["val_rmse"]}
        title = "Error (RMSE, g per person)"
    else:                    # older models: the root of the loss, an RMSE only for the mse loss
        curves = {"Training error": [math.sqrt(v) for v in hist["loss"]],
                  "Validation error": [math.sqrt(v) for v in hist["val_loss"]]}
        title = "Error (√ loss, g per person)"
    best = hist["val_loss"].index(min(hist["val_loss"])) + 1   # early stopping keeps the lowest validation loss
    data = pd.DataFrame([{"Epoch": i + 1, "Error": v * 1000, "Series": name}
                         for name, vals in curves.items() for i, v in enumerate(vals)])
    # Categorical slots 1-2 (blue, orange), stepped per theme; validated for CVD separation.
    dark = st.context.theme.type == "dark"
    color = alt.Color("Series:N", scale=alt.Scale(domain=list(curves), range=["#3987e5", "#d95926"] if dark
                                                  else ["#2a78d6", "#eb6834"]),
                      legend=alt.Legend(orient="top", title=None, labelFontSize=12))
    x = alt.X("Epoch:Q", scale=alt.Scale(domain=[1, max(n, 2)], nice=False),
              axis=alt.Axis(format="d", tickMinStep=1, title="Epoch"))
    y = alt.Y("Error:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format=".1f", title=title))
    tooltip = ["Series:N", "Epoch:Q", alt.Tooltip("Error:Q", format=".2f")]
    series = alt.Chart(data).encode(x=x, y=y, color=color)
    layers = [series.mark_line(strokeWidth=2),
              series.mark_point(filled=True, size=64, opacity=1).encode(tooltip=tooltip),
              series.mark_point(size=400, opacity=0).encode(tooltip=tooltip)]   # larger hit target
    layers.append(alt.Chart(pd.DataFrame({"Epoch": [best], "Error": [curves["Validation error"][best - 1] * 1000]}))
                  .mark_point(size=260, filled=False, strokeWidth=2, color="gray").encode(x=x, y=y))
    if n > best:
        # The epochs early stopping ran past the best one: none improved, which is the proof.
        checked = pd.DataFrame({"x": [best], "x2": [n], "mid": [(best + n) / 2],
                                "label": [f"{n - best} more epochs checked — none lower"]})
        same_x = dict(scale=alt.Scale(domain=[1, max(n, 2)], nice=False), title=None)
        layers.insert(0, alt.Chart(checked).mark_rect(opacity=0.12, color="gray")
                      .encode(x=alt.X("x:Q", **same_x), x2="x2:Q"))
        layers.append(alt.Chart(checked).mark_text(dy=8, baseline="top", color="gray", fontSize=12)
                      .encode(x=alt.X("mid:Q", **same_x), y=alt.value(0), text="label:N"))
    return alt.layer(*layers).properties(height=300)


@st.cache_data(show_spinner="Checking the current model…", max_entries=20)
def _live_evaluation(client_key, model_mtime, cases_mtime):
    """Every test-case dish and every measure for the model on disk; the mtimes re-run it after a retrain or new test cases."""
    return test_model.evaluate_current(client_key)


@st.cache_data(show_spinner="Reading the training data…", max_entries=20)
def _data_study(client_key, data_mtime, standards_mtime):
    """The training data by year; the mtimes re-run it when the dataset or the standard portions change."""
    return test_model.data_study(client_key)


def _mtime(path):
    return os.path.getmtime(path) if path and os.path.exists(path) else 0


def _render_model_test_section():
    """
    The model on disk, in five parts: 1 learning curve · 2 settings and the averaged models ·
    3 data study · 4 the last test set, day by day and dish by dish · 5 every measure on every set.
    Every error shown is measured now from the model files (cached until they or the test cases
    change); the learning curve, saved at training, is checked against them.
    """
    st.divider()
    st.markdown('<div class="sec"><span class="sec-icon">🧪</span><span class="sec-label">Model Test</span></div>', unsafe_allow_html=True)
    if st.button("Retrain model", key=key("retrain_btn"),
                 help="Retrain this client with the settings in model_params.json."):
        _retrain_from_ui()

    # ── 1. Learning curve ───────────────────────────────────────────────
    st.markdown("#### 1. Learning curve")
    hist = test_model.fit_history(CLIENT_KEY)
    if hist is None:
        st.info("No training history for this model yet — click **Retrain model**.")
        return
    verdict, explanation = test_model.fit_diagnosis(hist)
    (st.success if verdict == "good fit" else st.warning)(f"**{verdict.capitalize()}** — {explanation}")
    st.altair_chart(_fit_chart(hist), width="stretch")
    try:
        with predict_gate():
            ev = _live_evaluation(CLIENT_KEY, _mtime(os.path.join(ARTIFACT_DIR, f"{CLIENT_KEY}_per_pax_tf_model.keras")),
                                  _mtime(test_workbook_path(CLIENT_KEY)))
    except GateBusy as gate_busy:
        st.warning(str(gate_busy))
        return
    n_models = sum(r["Model"].startswith("Model ") for r in ev["models"])
    st.caption(f"Model 1{f' of {n_models}' if n_models > 1 else ''}. ◯ marks the epoch early stopping "
               "kept: the one with the lowest validation loss.")
    if "val_rmse" in hist:   # the curve's number at the kept epoch vs the same number measured now
        kept = hist["val_loss"].index(min(hist["val_loss"]))
        on_curve, now = hist["val_rmse"][kept] * 1000, ev["curve_check"] * 1000
        if abs(on_curve - now) < 0.05:
            st.caption(f"✓ In sync with the model on disk: model 1's raw validation RMSE at the kept epoch "
                       f"(before calibration and 5 g rounding) is {on_curve:.2f} g on the curve and {now:.2f} g "
                       "measured now.")
        else:
            st.warning(f"This curve does not match the model on disk: {on_curve:.2f} g on the curve, {now:.2f} g "
                       "measured now. It comes from another training run, or the data changed. Retrain.")
    else:
        st.caption("Retrain to check this curve against the model on disk.")

    # ── 2. Settings and the averaged models ─────────────────────────────
    st.markdown("#### 2. Settings and the averaged models")
    params = get_params(CLIENT_KEY)
    settings = [{"Setting": k, "Value": ", ".join(map(str, v)) if isinstance(v, list) else str(v),
                 "What it does": test_model.SETTING_NOTES.get(k, "")}
                for k, v in params.items() if not k.startswith("_")]
    settings.append({"Setting": "meal_day",
                     "Value": "from the menu" if getattr(LOGIC, "meal_day_from_menu", False) else "as logged",
                     "What it does": "veg, or the day's non-veg category (from the menu: the one picked above)"})
    st.dataframe(pd.DataFrame(settings), hide_index=True, width="stretch")
    models, meta = pd.DataFrame(ev["models"]), hist.get("models", [])   # meta: what only training knows
    for pos, (col, k) in enumerate((("Seed", "seed"), ("Epochs run", "epochs_run"), ("Best epoch", "best_epoch")), 1):
        models.insert(pos, col, [str(meta[i][k]) if i < len(meta) else "—" for i in range(len(models))])
    st.dataframe(models, hide_index=True, width="stretch", column_config={
        **{c: st.column_config.NumberColumn(format="%.1f") for c in ("Validation RMSE (g)", "Internal test RMSE (g)")},
        "Internal test within 5 g (%)": st.column_config.NumberColumn(format="%.1f%%")})
    st.caption("Errors measured now from the model files, as the app serves them (calibrated, rounded to 5 g) and "
               "scored like the overall table below, so the average's row matches it. That is why model 1 reads "
               "higher here than its raw value on the learning curve. Seed, epochs run and best epoch come from "
               "training" + ("." if meta else "; retrain to see them."))
    if params["rebase"] and os.path.exists(standards_path(CLIENT_KEY)):
        with st.expander("Standard portions used for re-basing"):
            st.dataframe(pd.read_csv(standards_path(CLIENT_KEY)), hide_index=True, width="stretch")

    # ── 3. Data study ───────────────────────────────────────────────────
    st.markdown("#### 3. Data study — the training data")
    summary, portions = _data_study(CLIENT_KEY, _mtime(CLIENT_INFO.get("dataset")), _mtime(standards_path(CLIENT_KEY)))
    st.dataframe(summary, hide_index=True, width="stretch")
    st.dataframe(portions, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%d") for c in portions.columns if c != "Category"})
    st.caption("Median per person, by year. Eaten = (cooked − wastage) ÷ client MG, so a dish with zero wastage "
               "shows eaten = cooked. The model learns eaten, re-based to today's standard when re-basing is on.")

    # ── 4. The last test set, day by day and dish by dish ───────────────
    cases = pd.DataFrame(ev["cases"])
    if "pred" not in cases:   # no dish could be predicted
        cases["pred"] = float("nan")
    labels = sorted(cases["test"].dropna().unique())
    name = labels[-1] if labels else "Test cases"
    shown = cases[cases["test"] == name] if labels else cases
    st.markdown(f"#### 4. {name} — day by day and dish by dish")
    st.caption(f"{shown['date'].min():%d %b} – {shown['date'].max():%d %b %Y}: {shown['date'].nunique()} days, "
               f"{len(shown)} dishes the model never trained on"
               + (f"; settings were chosen on {' and '.join(labels[:-1])}" if len(labels) > 1 else "") + ".")
    gap, mg = shown["pred"] - shown["actual"], pd.to_numeric(shown["mg"], errors="coerce")
    by_day = (shown.assign(short=(-gap).clip(lower=0) * mg, over=gap.clip(lower=0) * mg)
              .groupby("date").agg(**{"Day type": ("kind", "first"), "Dishes": ("dish", "size"),
                                      "Within 5 g": ("within_5g", "mean"), "Exact": ("exact", "sum"),
                                      "Within ±10%": ("within", "sum"), "kg short": ("short", "sum"),
                                      "kg over": ("over", "sum")}).reset_index())
    by_day.insert(4, "Target met", by_day["Within 5 g"] > test_model.DAY_TARGET)
    st.dataframe(by_day, hide_index=True, width="stretch", column_config={
        "date": st.column_config.DateColumn("Date", format="ddd D MMM"),
        "Within 5 g": st.column_config.ProgressColumn("Within 5 g", min_value=0, max_value=1, format="percent"),
        "Target met": st.column_config.CheckboxColumn(f"Over {test_model.DAY_TARGET:.0%}"),
        "kg short": st.column_config.NumberColumn(format="%.1f"),
        "kg over": st.column_config.NumberColumn(format="%.1f")})
    dishes = pd.DataFrame({
        "Date": shown["date"], "Dish": shown["dish"], "Category": shown["category"],
        "Predicted (g)": shown["pred"] * 1000, "Eaten (g)": shown["actual"] * 1000, "Difference (g)": gap * 1000,
        "Within 5 g": shown["within_5g"], "Within ±10%": shown["within"],
        "Note": [f"stand-in: {f}" if f else (e or "") for f, e in zip(shown["fallback"], shown["error"])]})
    if not dishes["Note"].any():
        dishes = dishes.drop(columns="Note")
    st.dataframe(dishes, hide_index=True, width="stretch", column_config={
        "Date": st.column_config.DateColumn(format="ddd D MMM"),
        "Predicted (g)": st.column_config.NumberColumn(format="%.0f"),
        "Eaten (g)": st.column_config.NumberColumn(format="%.0f"),
        "Difference (g)": st.column_config.NumberColumn(format="%+.0f")})

    # ── 5. Overall — every measure on every set ─────────────────────────
    st.markdown("#### 5. Overall — every measure on every set")
    st.dataframe(pd.DataFrame(ev["overall"]), hide_index=True, width="stretch", column_config={
        **{c: st.column_config.NumberColumn(format="%.1f%%") for c in ("Within 5 g (%)", "Exact (%)", "Within ±10% (%)")},
        **{c: st.column_config.NumberColumn(format="%.1f") for c in ("RMSE (g)", "MAE (g)", "kg short", "kg over")},
        "Bias (g)": st.column_config.NumberColumn(format="%+.1f")})
    st.caption(f"Predicted and eaten both on the app's 5 g grid. Within 5 g: within one serving-spoon step of what "
               f"was eaten per person; exact: the same step. Days over {test_model.DAY_TARGET:.0%}: days where more "
               f"than {test_model.DAY_TARGET:.0%} of the dishes were within 5 g. Shortage: predicted more than "
               f"{test_model.WITHIN_PCT:.0f}% below eaten. kg = the gap per person × client MG. The validation rows "
               "also set the calibration, so they flatter the model; validation and internal test are judged at "
               "today's standard portions.")


_render_model_test_section()
