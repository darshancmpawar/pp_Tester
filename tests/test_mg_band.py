"""The headcount input: client MG in bands, the same in training and in predict()."""
import ml_core


def test_band_edges_belong_to_the_lower_band():
    assert [ml_core.mg_band(m) for m in (40, 60, 61, 80, 81, 120, 121, 160, 161, 300)] == \
        ["0", "0", "1", "1", "2", "2", "3", "3", "4", "4"]
    assert ml_core.mg_band(None) == "0"
