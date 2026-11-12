"""Re-basing to today's standard portions, the kind of day, and the kitchen's dish-category fixes."""
import numpy as np
import pandas as pd
import pytest

import ml_core
from client_logic import get_logic


def test_rows_are_scaled_by_todays_standard_over_that_months_standard():
    df = pd.DataFrame({
        "category": ["dal", "dal", "healthy rice", "healthy rice", "curd"],
        "weekday":  ["monday", "tuesday", "wednesday", "monday", "monday"],
        "date":     pd.to_datetime(["2025-01-06", "2025-01-07", "2026-05-06", "2026-05-04", "2025-01-06"]),
        "per_pax":  [0.050, 0.050, 0.100, 0.050, 0.040],
        "ideal_pp": [0.050, 0.045, 0.100, 0.050, 0.040],
    })
    standards = pd.DataFrame({"category": ["Dal", "Healthy Rice", "Healthy Rice"],
                              "day": [np.nan, "Wednesday", np.nan], "grams": [70, 120, 60]})
    out = ml_core.rebase_to_standards(df, standards)["ideal_pp"].tolist()
    assert out == pytest.approx([0.070, 0.063,   # Jan 2025 dal at the old 50 g → ×1.4
                                 0.120,          # Wednesday healthy rice: its own standard
                                 0.060,          # other days: 60 g
                                 0.040])         # no standard: unchanged


def test_biryani_days_are_scaled_by_the_biryani_day_standard():
    dates = pd.to_datetime(["2026-03-04"] * 2 + ["2026-03-05"] * 2)
    df = pd.DataFrame({"category": ["white rice", "non-veg biryani", "white rice", "dal"],
                       "weekday": ["wednesday"] * 2 + ["thursday"] * 2, "date": dates,
                       "per_pax": [0.150, 0.350, 0.100, 0.070], "ideal_pp": [0.150, 0.350, 0.100, 0.070]})
    standards = pd.DataFrame({"category": ["white rice", "white rice"], "day": [np.nan, "Non-Veg Biryani"],
                              "grams": [100, 70]})
    out = ml_core.rebase_to_standards(df, standards)["ideal_pp"].tolist()
    assert out == pytest.approx([0.070, 0.350, 0.100, 0.070])   # biryani-day rice was 150 g then → 70 g today


def test_day_kind_is_the_non_veg_category_on_the_menu():
    assert ml_core.day_kind(["dal", "non-veg biryani", "white rice"]) == "non-veg biryani"
    assert ml_core.day_kind(["dal", "Non-Veg Gravy"]) == "non-veg gravy"
    assert ml_core.day_kind(["dal", "non-veg dry"]) == "non-veg dry"
    assert ml_core.day_kind(["non-veg gravy", "non-veg biryani"]) == "non-veg biryani"   # biryani first
    assert ml_core.day_kind(["dal", "veg gravy"]) == "veg"
    assert ml_core.day_kind(["dal", "non veg curry"]) == "non veg curry"   # Tekion's spelling


def test_kitchen_category_fixes_apply_to_predictions_too():
    fixes = get_logic("clario").dish_category_map
    assert fixes["gujarati kadi"][0] == "dal"
    assert fixes["bonda soup"][0] == "veg dry"
    assert fixes["alasande kalu sambar"][0] == "sambar/rasam"
