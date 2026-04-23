"""Purged + embargoed walk-forward cross-validation.

Why standard k-fold is wrong for time-series labels:
- Label at time t depends on the future (fwd_return_1d(t) = return from t to t+1).
- If train and test overlap in time, the model sees future info → leaked IC.

Walk-forward splits preserve time order. Purge + embargo further guard:
- **Purge**: when a test label at time t consumes info up to t+H (here H=1),
  training rows whose forward label overlaps that window must be excluded.
  For our 1-day horizon, purging just means "don't train on the test day
  itself" — trivial.
- **Embargo**: even after purge, short-horizon autocorrelation in features
  can leak. We drop N days around each test boundary from training.

Splitting scheme:
- **Expanding window** walk-forward: train = [start, test_start − embargo),
  test = [test_start, test_end]. Each fold uses progressively more history.
- `n_splits` splits the available date range into n equal-sized test blocks.

See López de Prado, "Advances in Financial Machine Learning," ch. 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Fold:
    fold_idx: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_dates: tuple[pd.Timestamp, pd.Timestamp]
    test_dates: tuple[pd.Timestamp, pd.Timestamp]


def purged_walk_forward(
    dates: pd.Series,
    n_splits: int = 6,
    embargo_days: int = 5,
    min_train_days: int = 252,
) -> Iterator[Fold]:
    """Yield walk-forward folds with train/test index arrays.

    `dates`: pandas Series of session_date per row (can be date or datetime).
              Must be aligned with the feature matrix (same row order).

    Split scheme:
    - Compute unique sorted session dates.
    - Reserve the first `min_train_days` as pure training (never tested).
    - Split remaining dates into `n_splits` equal test blocks.
    - For each test block, train on everything before `test_start - embargo`.
    """
    dates_ts = pd.to_datetime(dates).reset_index(drop=True)
    unique_dates = np.sort(dates_ts.unique())

    if len(unique_dates) < min_train_days + n_splits:
        raise ValueError(
            f"Not enough unique dates ({len(unique_dates)}) for "
            f"min_train_days={min_train_days} + n_splits={n_splits}"
        )

    # Available dates for testing (after warmup)
    test_range = unique_dates[min_train_days:]
    fold_size = len(test_range) // n_splits

    embargo = pd.Timedelta(days=embargo_days)

    for i in range(n_splits):
        t_start = test_range[i * fold_size]
        t_end = (
            test_range[(i + 1) * fold_size - 1]
            if i < n_splits - 1
            else test_range[-1]
        )
        t_start_ts = pd.Timestamp(t_start)
        t_end_ts = pd.Timestamp(t_end)
        train_cutoff = t_start_ts - embargo

        train_mask = dates_ts < train_cutoff
        test_mask = (dates_ts >= t_start_ts) & (dates_ts <= t_end_ts)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        train_min = dates_ts.iloc[train_idx].min()
        train_max = dates_ts.iloc[train_idx].max()
        yield Fold(
            fold_idx=i,
            train_idx=train_idx,
            test_idx=test_idx,
            train_dates=(train_min, train_max),
            test_dates=(t_start_ts, t_end_ts),
        )
