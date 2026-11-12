"""The co-menu: every dish is given the rest of its own day's menu, and nothing from other days."""
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
