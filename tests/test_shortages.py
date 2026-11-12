"""test_model.shortage_rows: which dishes count as a shortage, and how many kg short they are."""
import pandas as pd
import pytest

import test_model


def test_shortage_is_more_than_10pct_under_on_the_5g_grid_and_sized_in_kg():
    frame = pd.DataFrame({"date": ["2026-09-07"] * 3, "menu_items": ["dal", "rice", "rasam"],
                          "category": ["Dal", "White Rice", "Sambar/Rasam"], "client_mg": [100, 200, 50],
                          "actual": [0.070, 0.100, 0.060], "pred": [0.060, 0.090, 0.070]})
    n, out = test_model.shortage_rows(frame)
    assert n == 3
    assert list(out["dish"]) == ["dal"]                   # rice is exactly 10% under: within; rasam is over
    assert out["kg_short"].iloc[0] == pytest.approx(1.0)  # 10 g short × 100 heads
