"""ml_core.category_calibration: each category's median miss, shrunk toward 0 when it has few rows."""
import pytest

import ml_core


def test_calibration_is_the_median_miss_shrunk_by_row_count():
    resid = [0.01] * 10 + [-0.02] * 30 + [0.5]          # one wild dal row does not move the median
    cats = ["dal"] * 10 + ["white rice"] * 30 + ["dal"]
    cal = ml_core.category_calibration(resid, cats)
    assert cal["dal"] == pytest.approx(0.01 * 11 / 21)        # 11 rows → 11/21 of the median
    assert cal["white rice"] == pytest.approx(-0.02 * 30 / 40)
