"""test_model.efficiency: every measure the Model Test page shows for one set of dishes."""
import pandas as pd
import pytest

import test_model


def test_every_measure_on_two_small_days():
    frame = pd.DataFrame({
        "date": ["2026-10-13", "2026-10-13", "2026-10-14", "2026-10-14"],
        "dish": ["dal", "rasam", "white rice", "sambar"], "category": ["dal", "sambar/rasam", "white rice", "sambar/rasam"],
        "mg": [100] * 4,
        "actual": [0.100, 0.060, 0.100, 0.070],
        "pred":   [0.100, 0.065, 0.085, 0.070],   # exact, 5 g over, 15 g short (15%), exact
    })
    m = test_model.efficiency("two days", frame)
    assert (m["Dishes"], m["Days"]) == (4, 2)
    assert m["Within 5 g (%)"] == 75 and m["Exact (%)"] == 50 and m["Within ±10% (%)"] == 75
    assert m["Days over 70%"] == 1                     # day 1: 2 of 2; day 2: 1 of 2
    assert m["Shortages"] == 1 and m["kg short"] == pytest.approx(1.5) and m["kg over"] == pytest.approx(0.5)
    assert m["Bias (g)"] == pytest.approx(-2.5)
