"""The settings tab must read back exactly what was written (a duplicate row once crashed parse_settings)."""
import ml_core
import test_model
from client_logic import get_logic


def test_settings_tab_round_trips(tmp_path):
    L = get_logic("clario")
    params = {**ml_core.DEFAULT_PARAMS, "loss": "pseudo_huber", "loss_delta": 0.005,
              "dense_units": [64, 32], "mask_padding": True, "seed": 7}
    frame = test_model.settings_frame("clario", L, [("Config-1", "test", params)])
    assert not frame["parameter"].duplicated().any()
    path = str(tmp_path / "wb.xlsx")
    test_model.write_workbook(path, frame.iloc[:0], frame)
    [(name, label, back)] = test_model.parse_settings(path)
    assert (name, label, back) == ("Config-1", "test", params)
