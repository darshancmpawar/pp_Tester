"""
Unit tests for the model scorecard's two pure halves.

`evaluate_model` runs the Keras model and so needs the real artifacts; the
arithmetic it feeds (`scorecard`) and the split it reproduces (`_test_mask`)
are plain numpy/pandas and are what is tested here. The conftest stubs
tensorflow / sklearn / joblib, so this runs without them.

Run with:  pytest tests/test_scorecard.py -v
"""
import numpy as np
import pandas as pd
import pytest

import ml_core


def _frame(actual, predicted):
    """The shape evaluate_model returns."""
    df = pd.DataFrame({"actual": actual, "predicted": predicted})
    df["error"] = df["predicted"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    return df


class TestScorecard:
    def test_metrics_match_hand_arithmetic(self):
        card = ml_core.scorecard(
            _frame([0.10, 0.20, 0.00, 0.50], [0.12, 0.18, 0.05, 0.60])
        )
        # errors: +0.02, -0.02, +0.05, +0.10
        assert card["rows"] == 4
        assert card["rmse"] == pytest.approx(np.sqrt(0.013300 / 4))
        assert card["mae"] == pytest.approx(0.0475)
        assert card["bias"] == pytest.approx(0.0375)

    def test_mape_skips_zero_actual_rows_and_says_how_many(self):
        """Dividing by a zero actual is undefined; counting it as 0% flatters."""
        card = ml_core.scorecard(
            _frame([0.10, 0.20, 0.00, 0.50], [0.12, 0.18, 0.05, 0.60])
        )
        assert card["mape_rows"] == 3          # the zero-actual row is excluded
        assert card["rows"] == 4
        assert card["mape"] == pytest.approx((0.2 + 0.1 + 0.2) / 3 * 100)

    def test_bias_keeps_its_sign(self):
        """The whole point of bias: over-prediction and under-prediction differ."""
        over = ml_core.scorecard(_frame([0.1, 0.2], [0.15, 0.25]))
        under = ml_core.scorecard(_frame([0.1, 0.2], [0.05, 0.15]))
        assert over["bias"] > 0 and under["bias"] < 0
        # ...while RMSE cannot tell them apart. That is why both are reported.
        assert over["rmse"] == pytest.approx(under["rmse"])

    def test_empty_frame_is_rejected_not_averaged(self):
        with pytest.raises(ValueError):
            ml_core.scorecard(_frame([], []))


class TestTestMask:
    @staticmethod
    def _dated(n_dates, rows_per_date):
        dates = pd.to_datetime(["2026-01-%02d" % (d + 1) for d in range(n_dates)])
        return pd.DataFrame({"date": np.repeat(dates, rows_per_date)})

    def test_holds_out_whole_dates_never_part_of_one(self):
        """A row-level split leaks: co-menu context would see its own labels."""
        df = self._dated(10, 3)
        mask = ml_core._test_mask(df)
        train_dates = set(df.loc[~mask, "date"])
        test_dates = set(df.loc[mask, "date"])
        assert train_dates and test_dates
        assert not (train_dates & test_dates)

    def test_holds_out_about_test_size_of_the_dates(self):
        df = self._dated(10, 3)
        mask = ml_core._test_mask(df)
        assert df.loc[mask, "date"].nunique() == round(10 * ml_core.TEST_SIZE)

    def test_is_deterministic_so_training_and_scoring_agree(self):
        """The scorecard reproduces training's split from SEED alone."""
        df = self._dated(10, 3)
        assert np.array_equal(ml_core._test_mask(df), ml_core._test_mask(df))

    def test_single_date_dataset_still_splits_by_row(self):
        """Degenerate case: one service date. A split is still needed."""
        mask = ml_core._test_mask(self._dated(1, 20))
        assert 0 < mask.sum() < 20
