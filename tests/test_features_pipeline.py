"""Smoke tests for the feature pipeline on synthetic bars."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from bot8.features.quant.cross import rank_cross_sectional
from bot8.features.quant.stock import compute_stock_features


def _fake_bars(symbol: str, n_days: int = 300, seed: int = 0) -> pd.DataFrame:
    """Generate a clean synthetic bar series. Long enough for 252d features."""
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_days)]

    rets = rng.normal(0.0005, 0.015, n_days)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.002, n_days))
    high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
    low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
    volume = rng.integers(500_000, 2_000_000, n_days)

    return pd.DataFrame(
        {
            "symbol": symbol,
            "session_date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        }
    )


class TestStockFeatures:
    def test_output_shape(self) -> None:
        bars = _fake_bars("TEST", 300)
        out = compute_stock_features(bars)
        assert len(out) == len(bars)
        # Rough expected feature count — catches accidental additions/removals.
        added = out.shape[1] - bars.shape[1]
        assert added >= 50, f"expected >=50 features, got {added}"

    def test_warmup_nans(self) -> None:
        """Long-lookback features must be NaN during warmup period."""
        bars = _fake_bars("TEST", 300)
        out = compute_stock_features(bars)
        # ret_252d requires 252 priors, so row 100 must be NaN.
        assert pd.isna(out.loc[100, "ret_252d"])
        # Last rows should have it filled.
        assert not pd.isna(out.loc[270, "ret_252d"])

    def test_no_lookahead(self) -> None:
        """A change at bar t+k must not affect feature values at bar t<k.

        Computes features on the full series and on a truncated series up to
        day 200 — rows 0-150 must be identical."""
        bars = _fake_bars("TEST", 300)
        full = compute_stock_features(bars).iloc[:150]
        trunc = compute_stock_features(bars.iloc[:200]).iloc[:150]

        # Compare a sample of feature columns for exact equality (NaN-safe).
        for col in ["ret_20d", "rsi_14", "sma_20_ratio", "realized_vol_20d"]:
            pd.testing.assert_series_equal(
                full[col].reset_index(drop=True),
                trunc[col].reset_index(drop=True),
                check_exact=False,
                check_names=False,
                rtol=1e-10,
            )

    def test_rsi_bounded(self) -> None:
        out = compute_stock_features(_fake_bars("TEST", 300))
        rsi = out["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_consecutive_streaks_nonneg(self) -> None:
        out = compute_stock_features(_fake_bars("TEST", 100))
        assert (out["consecutive_up_days"] >= 0).all()
        assert (out["consecutive_down_days"] >= 0).all()


class TestCrossSectionalRanks:
    def test_each_day_ranks_in_0_1(self) -> None:
        """Per-date ranks must span [0, 1] with equal-spacing for clean ties."""
        dates = pd.date_range("2023-01-02", periods=5)
        wide = pd.DataFrame(
            [
                {"session_date": d.date(), "symbol": s, "ret_5d": v}
                for d in dates
                for s, v in zip("ABCD", [0.01, 0.02, 0.03, 0.04])
            ]
        )
        out = rank_cross_sectional(wide, ["ret_5d"])
        # Each date has 4 symbols; ranks should average (0.125, 0.375, 0.625, 0.875).
        per_day = out.groupby("session_date")["ret_5d_rank"].apply(
            lambda s: sorted(s.tolist())
        )
        for ranks in per_day:
            assert ranks == pytest.approx([0.25, 0.5, 0.75, 1.0])


import pytest  # noqa: E402 (fixture discovery)
