"""Re-basing to today's standard portions, and the kitchen's dish-category fixes."""
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
                              "weekday": [np.nan, "Wednesday", np.nan], "grams": [70, 120, 60]})
    out = ml_core.rebase_to_standards(df, standards)["ideal_pp"].tolist()
    assert out == pytest.approx([0.070, 0.063,   # Jan 2025 dal at the old 50 g → ×1.4
                                 0.120,          # Wednesday healthy rice: its own standard
                                 0.060,          # other days: 60 g
                                 0.040])         # no standard: unchanged


def test_kitchen_category_fixes_apply_to_predictions_too():
    fixes = get_logic("clario").dish_category_map
    assert fixes["gujarati kadi"][0] == "dal"
    assert fixes["bonda soup"][0] == "veg dry"
    assert fixes["alasande kalu sambar"][0] == "sambar/rasam"
