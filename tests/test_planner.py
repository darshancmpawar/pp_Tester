"""
Unit tests for planner.py — covers helpers, plan builders, and edge cases.

Run with:  pytest tests/test_planner.py -v
"""
import math
import pandas as pd
import pytest

from planner import (
    round_to_nearest_5 as mg5,
    ceil_to_next_5     as ceil5,
    round_to_005       as r005,
    vendor_pp_from_bump as vpp,
    avg_vendor_mg      as gavg,
    classify_rows      as classify,
    build_row,
    client_plan, fixed_pp_client_plan,
    vendor_plan, aggressive_plan, special_day_mg,
)


# ═══════════════════════════════════════════════════════════════════════
# MOCK LOGIC — implements just enough of the BaseLogic contract for planner
# ═══════════════════════════════════════════════════════════════════════
class MockLogic:
    star_categories = {"flavoured rice", "veg curry"}
    biryani_bump = 0.035
    default_bump = 0.01
    vendor_floor_ratio = 0.825
    vendor_mg_method = "tekion_2group"
    day_reductions = {
        "monday": 0.23, "tuesday": 0.21, "wednesday": 0.21,
        "thursday": 0.22, "friday": 0.23,
    }

    def adjust_vendor_mg(self, mg):
        if mg < 350:
            return mg * 0.95
        elif 350 <= mg <= 700:
            return mg - 15
        else:
            return mg - 25

    def adjust_nonveg_vendor_mg(self, mg):
        if mg < 400:
            return mg * 0.97
        else:
            return mg - 20

    def aggressive_bump(self, w):
        sp = 0.10 * w
        if sp <= 2:    f = 1.0
        elif sp <= 4:  f = 0.35
        elif sp <= 6:  f = 0.25
        elif sp <= 8:  f = 0.15
        else:          f = 0.10
        return round(w + f * sp, 1)

    def get_reduction_pct(self, dt, ht, wd):
        # mimic the matrix lookup; default to 10 like BaseLogic
        table = {
            ("holiday", "important holiday", "monday"): 12,
            ("holiday", "non-important holiday", "friday"): 8,
        }
        return table.get((dt.strip().lower(), ht.strip().lower(), wd.strip().lower()), 10)


class ThreeGroupLogic(MockLogic):
    vendor_mg_method = "3group"
    vendor_floor_ratio = 0.0

    def adjust_vendor_mg(self, mg):
        if mg < 120:
            return mg * 0.94
        else:
            return mg - 10


class DayBasedLogic(MockLogic):
    vendor_mg_method = "day_based"
    vendor_floor_ratio = 0.0


@pytest.fixture
def logic():
    return MockLogic()


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
class TestMg5:
    def test_zero(self):              assert mg5(0)    == 0
    def test_below_half(self):        assert mg5(2)    == 0
    def test_above_half(self):        assert mg5(3)    == 5
    def test_exact_multiple(self):    assert mg5(15)   == 15
    def test_floats(self):            assert mg5(7.4)  == 5
    def test_large(self):             assert mg5(287)  == 285


class TestCeil5:
    def test_zero(self):              assert ceil5(0)     == 0
    def test_below(self):             assert ceil5(1)     == 5
    def test_exact(self):             assert ceil5(5)     == 5
    def test_just_over(self):         assert ceil5(5.01)  == 10
    def test_six(self):               assert ceil5(6)     == 10


class TestR005:
    def test_zero(self):              assert r005(0)      == 0
    def test_below_threshold(self):   assert r005(0.062)  == 0.06
    def test_above_threshold(self):   assert r005(0.063)  == 0.065
    def test_exact(self):             assert r005(0.075)  == 0.075


class TestVpp:
    """Vendor PP is ceil((pp + bump) * 100) / 100."""

    def test_basic_floating_point_quirk(self):
        # Mathematically (0.05 + 0.01) * 100 = 6.0, but in IEEE-754 it lands on
        # 6.000000000000001 and math.ceil rounds *up* one extra cent.
        # This is the *actual* behavior of vpp — codifying it so any future
        # rewrite that "fixes" this catches a regression here first.
        assert vpp(0.05, 0.01) == 0.07

    def test_clean_ceil(self):
        # 0.052 + 0.01 = 0.062 → ceil(6.2)/100 = 0.07
        assert vpp(0.052, 0.01) == 0.07

    def test_with_biryani_bump(self):
        # 0.20 + 0.035 = 0.235 → ceil(23.5)/100 = 0.24
        assert vpp(0.20, 0.035) == 0.24

    def test_already_at_cent(self):
        # Same float drift: 0.10 + 0.02 → 0.12000…1 → ceil bumps to 0.13
        assert vpp(0.10, 0.02) == 0.13


class TestGavg:
    def test_empty_returns_zero(self):
        assert gavg([]) == 0

    def test_single_entry(self):
        assert gavg([{"Vendor MG": 100}]) == 100

    def test_multiple(self):
        assert gavg([{"Vendor MG": 100}, {"Vendor MG": 200}]) == 150

    def test_custom_key(self):
        assert gavg([{"Total Qty": 10}, {"Total Qty": 20}], key="Total Qty") == 15


# ═══════════════════════════════════════════════════════════════════════
# CLASSIFY
# ═══════════════════════════════════════════════════════════════════════
class TestClassify:
    def _rows(self):
        return [
            {"Category": "non veg curry",  "Vendor MG": 50},
            {"Category": "flavoured rice", "Vendor MG": 200},
            {"Category": "veg curry",      "Vendor MG": 250},
            {"Category": "indian bread",   "Vendor MG": 300},
            {"Category": "dal",            "Vendor MG": 350},
        ]

    def test_basic_split(self, logic):
        nv, st_, rest = classify(self._rows(), "non veg curry", logic.star_categories)
        assert len(nv) == 1
        assert len(st_) == 2
        assert len(rest) == 2

    def test_no_nv_cat(self, logic):
        nv, st_, rest = classify(self._rows(), None, logic.star_categories)
        assert nv == []
        assert len(st_) + len(rest) == 5

    def test_empty_results(self, logic):
        nv, st_, rest = classify([], "x", logic.star_categories)
        assert nv == [] and st_ == [] and rest == []

    def test_case_insensitive(self, logic):
        rows = [{"Category": "NON VEG CURRY", "Vendor MG": 1}]
        nv, _, _ = classify(rows, "non veg curry", logic.star_categories)
        assert len(nv) == 1


# ═══════════════════════════════════════════════════════════════════════
# BUILD_ROW
# ═══════════════════════════════════════════════════════════════════════
class TestBuildRow:
    def test_default_bump(self, logic):
        row = build_row(0.05, 15.0, "veg dry", "potato fry", False, None, logic)
        assert row["Client PP"] == 0.05
        # vpp float-drift quirk: ceil((0.05+0.01)*100)/100 → 0.07 not 0.06
        assert row["Vendor PP"] == 0.07
        assert row["Total Qty"] == 15.0
        assert row["Item"] == "potato fry"
        assert row["Category"] == "veg dry"
        # vendor mg = total_qty / vendor_pp = 15 / 0.07 ≈ 214.29
        assert row["Vendor MG"] == pytest.approx(15.0 / 0.07)

    def test_biryani_bump_applied(self, logic):
        # is_nv_day=True, nv_cat matches and is biryani → biryani_bump used
        row = build_row(0.20, 50.0, "non veg biryani", "chicken biryani",
                        True, "non veg biryani", logic)
        # vpp = ceil((0.20+0.035)*100)/100 = 0.24
        assert row["Vendor PP"] == 0.24

    def test_biryani_bump_not_triggered_when_not_nv_day(self, logic):
        # is_nv_day=False → default bump even if cat is biryani
        row = build_row(0.20, 50.0, "non veg biryani", "x",
                        False, "non veg biryani", logic)
        # vpp float-drift: ceil((0.20+0.01)*100)/100 → 0.22 not 0.21
        assert row["Vendor PP"] == 0.22

    def test_vendor_mg_protects_against_zero_pp(self, logic):
        # build_row uses max(vp, 1e-9) to prevent div by zero
        row = build_row(0, 0, "salad", "x", False, None, logic)
        # vp = ceil((0+0.01)*100)/100 = 0.01, never 0
        assert row["Vendor PP"] > 0


# ═══════════════════════════════════════════════════════════════════════
# CLIENT PLAN
# ═══════════════════════════════════════════════════════════════════════
class TestClientPlan:
    def test_columns_subset(self):
        df = pd.DataFrame([{
            "Category": "veg dry", "Item": "x", "Client PP": 0.05,
            "Vendor PP": 0.06, "Total Qty": 15, "Vendor MG": 250,
        }])
        cp = client_plan(df)
        assert list(cp.columns) == ["Category", "Item", "Client PP", "Total Qty"]
        assert "Vendor PP" not in cp.columns
        assert "Vendor MG" not in cp.columns


# ═══════════════════════════════════════════════════════════════════════
# FIXED PP CLIENT PLAN
# ═══════════════════════════════════════════════════════════════════════
class TestFixedPpClientPlan:
    def test_uses_fixed_pp(self):
        df = pd.DataFrame([
            {"Category": "veg dry",       "Item": "potato"},
            {"Category": "indian bread",  "Item": "roti"},
        ])
        fpp = {"veg dry": 0.075, "indian bread": 0.05}
        cp = fixed_pp_client_plan(df, fpp, meal_group=200)
        assert cp.iloc[0]["Client PP"] == 0.075
        assert cp.iloc[0]["Total Qty"] == 15.0          # 0.075 * 200
        assert cp.iloc[1]["Client PP"] == 0.05
        assert cp.iloc[1]["Total Qty"] == 10.0

    def test_unknown_category_defaults(self):
        df = pd.DataFrame([{"Category": "mystery cat", "Item": "x"}])
        cp = fixed_pp_client_plan(df, {"veg dry": 0.075}, meal_group=100)
        # unknown → defaults to 0.10
        assert cp.iloc[0]["Client PP"] == 0.10
        assert cp.iloc[0]["Total Qty"] == 10.0


# ═══════════════════════════════════════════════════════════════════════
# VENDOR PLAN — TEKION 2-GROUP
# ═══════════════════════════════════════════════════════════════════════
class TestVendorPlanTekion:
    def _build(self, logic):
        df_rows = [
            {"Category": "flavoured rice", "Item": "veg pulao",
             "Client PP": 0.10, "Vendor PP": 0.11, "Total Qty": 30, "Vendor MG": 273},
            {"Category": "veg curry", "Item": "paneer",
             "Client PP": 0.10, "Vendor PP": 0.11, "Total Qty": 30, "Vendor MG": 273},
            {"Category": "indian bread", "Item": "roti",
             "Client PP": 0.05, "Vendor PP": 0.06, "Total Qty": 15, "Vendor MG": 250},
            {"Category": "dal", "Item": "dal tadka",
             "Client PP": 0.066, "Vendor PP": 0.08, "Total Qty": 19.8, "Vendor MG": 247},
        ]
        df = pd.DataFrame(df_rows)
        results = df.to_dict("records")
        return df, results

    def test_veg_only_basic(self, logic):
        df, results = self._build(logic)
        vp, vmg, nvmg = vendor_plan(df, results, client_mg=300, is_nonveg_day=False, nonveg_cat=None,
                                    logic=logic, weekday="monday")
        assert nvmg == 0                       # no NV
        assert vmg > 0
        assert vmg % 5 == 0                    # rounded to nearest 5
        assert "Ordered Qty" in vp.columns

    def test_floor_ratio_enforced(self, logic):
        # If raw VMG falls below floor ratio * cmg, ceil5 lift kicks in
        df, results = self._build(logic)
        # cmg high enough that 0.825 * cmg > the computed vmg → floor lifts it
        vp, vmg, nvmg = vendor_plan(df, results, client_mg=400, is_nonveg_day=False, nonveg_cat=None,
                                    logic=logic, weekday="monday")
        assert vmg + nvmg >= 0.825 * 400

    def test_ordered_qty_never_below_total_qty(self, logic):
        df, results = self._build(logic)
        vp, _, _ = vendor_plan(df, results, client_mg=300, is_nonveg_day=False, nonveg_cat=None,
                               logic=logic, weekday="monday")
        for _, row in vp.iterrows():
            orig = df[df["Item"] == row["Item"]]["Total Qty"].iloc[0]
            assert row["Ordered Qty"] >= orig

    def test_with_nonveg_separate_track(self, logic):
        df_rows = [
            {"Category": "flavoured rice", "Item": "x",
             "Client PP": 0.10, "Vendor PP": 0.11, "Total Qty": 30, "Vendor MG": 273},
            {"Category": "non veg curry", "Item": "chicken",
             "Client PP": 0.12, "Vendor PP": 0.13, "Total Qty": 12, "Vendor MG": 92},
        ]
        df = pd.DataFrame(df_rows)
        vp, vmg, nvmg = vendor_plan(df, df.to_dict("records"), client_mg=100,
                                    is_nonveg_day=True, nonveg_cat="non veg curry",
                                    logic=logic, weekday="tuesday")
        assert nvmg > 0    # NV track populated


# ═══════════════════════════════════════════════════════════════════════
# VENDOR PLAN — 3-GROUP
# ═══════════════════════════════════════════════════════════════════════
class TestVendorPlan3Group:
    def test_basic(self):
        logic = ThreeGroupLogic()
        df = pd.DataFrame([
            {"Category": "flavoured rice", "Item": "x",
             "Client PP": 0.10, "Vendor PP": 0.11, "Total Qty": 30, "Vendor MG": 273},
            {"Category": "indian bread", "Item": "y",
             "Client PP": 0.05, "Vendor PP": 0.06, "Total Qty": 15, "Vendor MG": 250},
            {"Category": "non veg curry", "Item": "z",
             "Client PP": 0.12, "Vendor PP": 0.13, "Total Qty": 12, "Vendor MG": 92},
        ])
        vp, vmg, nvmg = vendor_plan(df, df.to_dict("records"), client_mg=200,
                                    is_nonveg_day=True, nonveg_cat="non veg curry",
                                    logic=logic, weekday="monday")
        assert nvmg == 0          # 3group always reports nvmg = 0 (single track)
        assert vmg > 0
        assert vmg % 5 == 0


# ═══════════════════════════════════════════════════════════════════════
# VENDOR PLAN — DAY BASED
# ═══════════════════════════════════════════════════════════════════════
class TestVendorPlanDayBased:
    def test_known_weekday(self):
        logic = DayBasedLogic()
        df = pd.DataFrame([{
            "Category": "veg dry", "Item": "x",
            "Client PP": 0.05, "Vendor PP": 0.06, "Total Qty": 15, "Vendor MG": 250,
        }])
        # Monday: 0.23 reduction → 200*(1-0.23) = 154 → mg5 = 155
        vp, vmg, nvmg = vendor_plan(df, df.to_dict("records"), client_mg=200,
                                    is_nonveg_day=False, nonveg_cat=None,
                                    logic=logic, weekday="monday")
        assert vmg == 155
        assert nvmg == 0

    def test_unknown_weekday_uses_default(self):
        logic = DayBasedLogic()
        df = pd.DataFrame([{
            "Category": "veg dry", "Item": "x",
            "Client PP": 0.05, "Vendor PP": 0.06, "Total Qty": 15, "Vendor MG": 250,
        }])
        vp, vmg, _ = vendor_plan(df, df.to_dict("records"), client_mg=200,
                                 is_nonveg_day=False, nonveg_cat=None,
                                 logic=logic, weekday="saturday")  # not in dict
        # fallback 0.22: 200*0.78 = 156 → mg5 = 155
        assert vmg == 155

    def test_zero_protection(self):
        """Day-based VMG should never collapse to 0 even with extreme inputs."""
        logic = DayBasedLogic()
        df = pd.DataFrame([{
            "Category": "veg dry", "Item": "x",
            "Client PP": 0.05, "Vendor PP": 0.06, "Total Qty": 1, "Vendor MG": 17,
        }])
        # cmg=1 → after reduction this becomes very small; planner enforces vmg >= 5
        vp, vmg, _ = vendor_plan(df, df.to_dict("records"), client_mg=1,
                                 is_nonveg_day=False, nonveg_cat=None,
                                 logic=logic, weekday="monday")
        assert vmg >= 5
        # Vendor PP must not be inf/nan
        for v in vp["Vendor PP"]:
            assert math.isfinite(v)
            assert v > 0

    def test_ordered_qty_floors_at_total_qty(self):
        logic = DayBasedLogic()
        df = pd.DataFrame([{
            "Category": "veg dry", "Item": "x",
            "Client PP": 0.05, "Vendor PP": 0.06, "Total Qty": 15.0, "Vendor MG": 250,
        }])
        vp, _, _ = vendor_plan(df, df.to_dict("records"), client_mg=200,
                               is_nonveg_day=False, nonveg_cat=None,
                               logic=logic, weekday="monday")
        assert vp.iloc[0]["Ordered Qty"] >= 15.0


# ═══════════════════════════════════════════════════════════════════════
# AGGRESSIVE PLAN
# ═══════════════════════════════════════════════════════════════════════
class TestAggressivePlan:
    def _vp_results(self):
        rows = [
            {"Category": "flavoured rice", "Item": "pulao",
             "Vendor PP": 0.11, "Ordered Qty": 33.0},
            {"Category": "indian bread",   "Item": "roti",
             "Vendor PP": 0.06, "Ordered Qty": 18.0},
            {"Category": "dal",            "Item": "dal",
             "Vendor PP": 0.08, "Ordered Qty": 24.0},
        ]
        return pd.DataFrame(rows), [
            {**r, "Total Qty": r["Ordered Qty"], "Vendor MG": 300} for r in rows
        ]

    def test_star_categories_get_bumped(self, logic):
        vp, results = self._vp_results()
        ag, *_ = aggressive_plan(vp, results, final_veg_mg=300, avg_nonveg_mg=0,
                                  is_nonveg_day=False, nonveg_cat=None, logic=logic, method_groups=2)
        # Star (flavoured rice) row should be bumped up
        star_row = ag[ag["Category"] == "flavoured rice"].iloc[0]
        non_star_row = ag[ag["Category"] == "indian bread"].iloc[0]
        assert star_row["Ordered Qty"] > 33.0
        assert non_star_row["Ordered Qty"] == 18.0

    def test_outputs_are_mg5_aligned(self, logic):
        vp, results = self._vp_results()
        _, adj_t, adj_nv, adj_v = aggressive_plan(
            vp, results, final_veg_mg=300, avg_nonveg_mg=0,
            is_nonveg_day=False, nonveg_cat=None, logic=logic, method_groups=2,
        )
        assert adj_t % 5 == 0
        assert adj_nv % 5 == 0
        assert adj_v % 5 == 0

    def test_negative_clamped_to_zero(self, logic):
        vp, results = self._vp_results()
        _, adj_t, adj_nv, adj_v = aggressive_plan(
            vp, results, final_veg_mg=-50, avg_nonveg_mg=-100,
            is_nonveg_day=False, nonveg_cat=None, logic=logic, method_groups=2,
        )
        assert adj_t >= 0 and adj_nv >= 0 and adj_v >= 0

    def test_adj_nv_never_exceeds_adj_t(self, logic):
        """Safety: if NV ends up > total, planner clamps NV down to total."""
        vp = pd.DataFrame([
            {"Category": "non veg curry", "Item": "x", "Vendor PP": 0.13, "Ordered Qty": 50.0},
        ])
        results = [{"Category": "non veg curry", "Vendor PP": 0.13,
                    "Ordered Qty": 50.0, "Total Qty": 50.0, "Vendor MG": 385}]
        ag, adj_t, adj_nv, adj_v = aggressive_plan(
            vp, results, final_veg_mg=10, avg_nonveg_mg=400,
            is_nonveg_day=True, nonveg_cat="non veg curry", logic=logic, method_groups=2,
        )
        assert adj_nv <= adj_t
        assert adj_v == adj_t - adj_nv

    def test_3group_path(self):
        logic = ThreeGroupLogic()
        vp, results = TestAggressivePlan()._vp_results()
        _, adj_t, adj_nv, _ = aggressive_plan(
            vp, results, final_veg_mg=200, avg_nonveg_mg=0,
            is_nonveg_day=False, nonveg_cat=None, logic=logic, method_groups=3,
        )
        # 3-group never produces a separate nv MG
        assert adj_nv == 0

    def test_zero_vendor_pp_does_not_crash(self, logic):
        """If Vendor PP is 0, AMG calculation must use safe division."""
        vp = pd.DataFrame([
            {"Category": "veg dry", "Item": "x", "Vendor PP": 0.0, "Ordered Qty": 10.0},
        ])
        results = [{"Category": "veg dry", "Vendor PP": 0.0, "Ordered Qty": 10.0,
                    "Total Qty": 10.0, "Vendor MG": 0}]
        # Should not raise
        aggressive_plan(vp, results, final_veg_mg=100, avg_nonveg_mg=0,
                        is_nonveg_day=False, nonveg_cat=None, logic=logic, method_groups=2)


# ═══════════════════════════════════════════════════════════════════════
# SPECIAL DAY MG
# ═══════════════════════════════════════════════════════════════════════
class TestSpecialDayMg:
    def test_known_combo(self, logic):
        # ("holiday", "important holiday", "monday") → 12% reduction
        vmg, pct = special_day_mg(300, "holiday", "important holiday", "monday", logic)
        assert pct == 12
        assert vmg == pytest.approx(300 * 0.88)

    def test_unknown_combo_defaults_to_10(self, logic):
        vmg, pct = special_day_mg(300, "regular", "not applicable", "monday", logic)
        assert pct == 10
        assert vmg == pytest.approx(270.0)

    def test_zero_cmg(self, logic):
        vmg, pct = special_day_mg(0, "holiday", "important holiday", "monday", logic)
        assert vmg == 0

    def test_case_handling(self, logic):
        # logic.get_reduction_pct lowercases internally
        vmg1, pct1 = special_day_mg(100, "Holiday", "Important Holiday", "Monday", logic)
        vmg2, pct2 = special_day_mg(100, "holiday", "important holiday", "monday", logic)
        assert pct1 == pct2
        assert vmg1 == vmg2
