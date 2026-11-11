"""scripts/search.py: the kitchen cost and the keep rule decide which model wins."""
import pytest

import scripts.search as search


def test_kitchen_cost_scales_with_headcount_and_weights_shortage():
    # 10 g short for 40 heads = 0.4 kg × weight; 10 g over for 250 heads = 2.5 kg
    assert search.mg_cost([0.05, 0.11], [0.06, 0.10], [40, 250], weight=1.5) == pytest.approx(0.4 * 1.5 + 2.5)
    assert search.mg_cost([0.05, 0.11], [0.06, 0.10], [40, 250], weight=1.5, power=0) == pytest.approx(0.01 * 1.5 + 0.01)
    assert search.mg_cost([0.10], [0.10], [200], weight=2) == 0


def test_keep_rule_needs_5g_gain_no_10pct_loss_and_lower_cost_everywhere():
    costs = lambda v: {f"cost {k} @{search.SHORTAGE_WEIGHT}": v for k in search.MG_POWERS}
    base = {"w5": 100, "within": 110, "rmse": 0.0136, **costs(50.0)}
    good = {"w5": 104, "within": 110, "rmse": 0.0137, **costs(48.0)}
    assert search.keeps(good, base)
    assert not search.keeps({**good, "w5": 103}, base)                        # too few more within 5 g
    assert not search.keeps({**good, "within": 109}, base)                    # lost a dish within ±10%
    assert not search.keeps({**good, **costs(50.0)}, base)                    # cost not lower
    assert not search.keeps({**good, "rmse": 0.0150}, base)                   # internal error worse
    assert not search.keeps({**good, f"cost kg (MG) @{search.SHORTAGE_WEIGHT}": 49.9}, base)   # one scaling not lower
