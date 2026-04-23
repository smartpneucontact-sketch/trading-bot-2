"""Tests for purged walk-forward CV splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot8.models.quant.cv import purged_walk_forward


def _date_series(n_days: int, n_symbols: int = 10) -> pd.Series:
    """Generate a dates Series simulating n_days of history × n_symbols cross-section."""
    start = pd.Timestamp("2019-01-01")
    dates = [start + pd.Timedelta(days=i) for i in range(n_days)]
    # Duplicate each date n_symbols times (simulating cross-section)
    return pd.Series(np.repeat(dates, n_symbols))


class TestWalkForward:
    def test_produces_expected_number_of_folds(self) -> None:
        dates = _date_series(n_days=500)
        folds = list(purged_walk_forward(dates, n_splits=4, min_train_days=100, embargo_days=5))
        assert len(folds) == 4

    def test_no_overlap_between_train_and_test(self) -> None:
        dates = _date_series(n_days=500)
        for fold in purged_walk_forward(dates, n_splits=4, min_train_days=100, embargo_days=5):
            train_dates = set(pd.to_datetime(dates.iloc[fold.train_idx]).unique())
            test_dates = set(pd.to_datetime(dates.iloc[fold.test_idx]).unique())
            assert not (train_dates & test_dates), (
                f"fold {fold.fold_idx}: train and test share dates"
            )

    def test_embargo_enforced(self) -> None:
        """Latest training date must be at least embargo_days before earliest test date."""
        dates = _date_series(n_days=500)
        embargo = 5
        for fold in purged_walk_forward(
            dates, n_splits=4, min_train_days=100, embargo_days=embargo
        ):
            train_max = pd.to_datetime(dates.iloc[fold.train_idx]).max()
            test_min = pd.to_datetime(dates.iloc[fold.test_idx]).min()
            gap = (test_min - train_max).days
            assert gap >= embargo, f"embargo violated: {gap} < {embargo}"

    def test_walk_forward_order(self) -> None:
        """Test dates should be strictly increasing across folds."""
        dates = _date_series(n_days=800)
        folds = list(purged_walk_forward(dates, n_splits=5, min_train_days=200, embargo_days=5))
        last_test_start = pd.Timestamp.min
        for fold in folds:
            t_start = fold.test_dates[0]
            assert t_start > last_test_start, "test folds must advance in time"
            last_test_start = t_start

    def test_insufficient_data_raises(self) -> None:
        dates = _date_series(n_days=50)  # not enough
        with pytest.raises(ValueError, match="Not enough"):
            list(purged_walk_forward(dates, n_splits=4, min_train_days=100, embargo_days=5))


class TestMetrics:
    def test_per_day_ic_known(self) -> None:
        """If preds exactly match returns, per-day IC should be 1.0 for each day."""
        from bot8.models.quant.metrics import per_day_ic

        n_days = 10
        n_sym = 20
        dates = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=n_days), n_sym))
        y_true = pd.Series(np.random.default_rng(0).standard_normal(n_days * n_sym))
        y_pred = y_true.copy()  # perfect prediction
        ic = per_day_ic(y_true, y_pred, dates).dropna()
        assert (ic > 0.99).all()

    def test_signal_stats_zero_for_random(self) -> None:
        from bot8.models.quant.metrics import signal_stats

        rng = np.random.default_rng(1)
        n_days, n_sym = 60, 50
        dates = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=n_days), n_sym))
        y_true = pd.Series(rng.standard_normal(n_days * n_sym))
        y_pred = pd.Series(rng.standard_normal(n_days * n_sym))
        stats = signal_stats(y_true, y_pred, dates)
        assert abs(stats.ic_mean) < 0.1
