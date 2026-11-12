"""The co-menu: every dish is given the rest of its own day's menu, and nothing from other days."""
import numpy as np
import pandas as pd

import ml_core


def test_every_dish_is_given_its_whole_day_menu_in_training():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-01"] * 3 + ["2026-09-02"] * 2),
        "menu_items": ["dal tadka", "jeera rice", "chicken biryani", "dal tadka", "tomato rasam"],
        "category": ["dal", "flavoured rice", "non-veg biryani", "dal", "sambar/rasam"],
        "ideal_pp": [0.07, 0.15, 0.35, 0.07, 0.06],
    })
    columns = ["menu_items", "category"]
    encoders = ml_core.fit_encoders(df, columns)
    X, _ = ml_core.build_inputs(df, columns, encoders)
    item = {name: int(encoders["menu_items"].transform([name])[0]) for name in df["menu_items"]}
    pad = len(encoders["menu_items"].classes_)
    ctx = [set(int(i) for i in row if i != pad) for row in X[1]]
    assert ctx[0] == {item["jeera rice"], item["chicken biryani"]}   # dal on 1 Sep sees the other two
    assert ctx[3] == {item["tomato rasam"]}                           # dal on 2 Sep sees only rasam


def test_a_menu_with_no_other_dishes_averages_to_zero_not_nan():
    emb = np.array([[[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]], [[9.0, 9.0], [9.0, 9.0], [9.0, 9.0]]])
    ids = np.array([[0, 1, 5], [5, 5, 5]])   # 5 = the pad slot
    out = np.asarray(ml_core.masked_mean(emb, ids, 5))
    assert out.tolist() == [[2.0, 3.0], [0.0, 0.0]]
