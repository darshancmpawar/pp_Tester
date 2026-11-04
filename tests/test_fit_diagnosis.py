"""fit_diagnosis must not call an early-stopped model "overfitting": it keeps the best epoch's weights."""
import test_model


def _hist(val, max_epochs, early_stopping):
    loss = [0.01 / (i + 1) for i in range(len(val))]          # training loss keeps falling
    return {"loss": loss, "val_loss": val, "target_std": 0.1,
            "early_stopping": early_stopping, "max_epochs": max_epochs}


def test_early_stopped_run_is_a_good_fit():
    val = [0.004, 0.002, 0.001, 0.0013, 0.0014, 0.0015]       # bottoms at epoch 3, then rises
    verdict, text = test_model.fit_diagnosis(_hist(val, max_epochs=50, early_stopping=3))
    assert verdict == "good fit"
    assert "kept the weights from epoch 3" in text


def test_same_curve_without_early_stopping_is_overfitting():
    val = [0.004, 0.002, 0.001, 0.0013, 0.0014, 0.0015]
    verdict, _ = test_model.fit_diagnosis(_hist(val, max_epochs=6, early_stopping=0))
    assert verdict == "overfitting"
