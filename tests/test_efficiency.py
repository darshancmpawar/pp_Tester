"""test_model.efficiency: every measure the Model Test page shows for one set of dishes."""
import pandas as pd
import pytest

import test_model


def test_every_measure_on_two_small_days():
    frame = pd.DataFrame({
        "date": ["2026-10-13", "2026-10-13", "2026-10-14", "2026-10-14"],
        "actual": [0.100, 0.060, 0.100, 0.070],
        "pred":   [0.100, 0.065, 0.085, 0.070],   # exact, 5 g over, 15 g short (15%), exact
    })
    m = test_model.efficiency("two days", frame)
    assert (m["Dishes"], m["Days"]) == (4, 2)
    assert m["Within 5 g (%)"] == 75 and m["Exact (%)"] == 50 and m["Within ±10% (%)"] == 75
    assert m["Days over 70%"] == 1                     # day 1: 2 of 2; day 2: 1 of 2
    assert (m["Short"], m["Over"]) == (1, 0)           # 5 g on 60 g is 8.3% over: within
    assert m["Bias (pp)"] == pytest.approx(-0.0025)


def test_exactly_10_percent_off_is_within_not_short():
    frame = pd.DataFrame({"date": ["2026-09-07"] * 2, "actual": [0.100, 0.070], "pred": [0.090, 0.060]})
    m = test_model.efficiency("edge", frame)
    assert m["Within ±10% (%)"] == 50                  # 90 vs 100 is exactly 10%: within
    assert m["Short"] == 1                             # 60 vs 70 is 14% under: short
