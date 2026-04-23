"""Unit tests for technical indicator functions.

Strategy: use known inputs where the expected output is computable by hand
or from a reference formula. We're not testing against TA-lib (different
smoothing) — just verifying our definitions are internally consistent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot8.features.quant import technicals as T


@pytest.fixture
def trending_up() -> pd.DataFrame:
    """A steadily rising series with a small wiggle — realistic for testing."""
    n = 100
    trend = np.linspace(100, 150, n)
    noise = np.sin(np.arange(n) / 3) * 0.5
    close = pd.Series(trend + noise)
    high = close + 0.5
    low = close - 0.5
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.ones(n) * 1_000_000)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


class TestRSI:
    def test_bounded(self, trending_up: pd.DataFrame) -> None:
        rsi = T.rsi(trending_up["close"], 14).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_high_for_uptrend(self, trending_up: pd.DataFrame) -> None:
        rsi = T.rsi(trending_up["close"], 14).dropna()
        # Monotonic-ish uptrend should show RSI consistently above 50
        assert rsi.mean() > 60


class TestMACD:
    def test_hist_equals_line_minus_signal(self, trending_up: pd.DataFrame) -> None:
        line, sig, hist = T.macd(trending_up["close"])
        diff = (hist - (line - sig)).dropna().abs().max()
        assert diff < 1e-10


class TestBollinger:
    def test_close_lies_between_bands_mostly(self, trending_up: pd.DataFrame) -> None:
        upper, _, lower, _, _ = T.bollinger(trending_up["close"], 20, 2.0)
        df = pd.DataFrame({"c": trending_up["close"], "u": upper, "l": lower}).dropna()
        inside = ((df["c"] <= df["u"]) & (df["c"] >= df["l"])).mean()
        # ~95% expected for normal noise; loose bound for our sine wiggle.
        assert inside > 0.85

    def test_pct_b_in_zero_one_mostly(self, trending_up: pd.DataFrame) -> None:
        _, _, _, pct_b, _ = T.bollinger(trending_up["close"], 20, 2.0)
        pb = pct_b.dropna()
        assert (pb >= -0.5).mean() > 0.98  # should rarely go far negative


class TestATR:
    def test_positive(self, trending_up: pd.DataFrame) -> None:
        a = T.atr(trending_up["high"], trending_up["low"], trending_up["close"], 14)
        a = a.dropna()
        assert (a > 0).all()


class TestStochastic:
    def test_bounded(self, trending_up: pd.DataFrame) -> None:
        k, d = T.stochastic(
            trending_up["high"], trending_up["low"], trending_up["close"], 14, 3
        )
        for s in (k.dropna(), d.dropna()):
            assert (s >= 0).all() and (s <= 100).all()


class TestWilliamsR:
    def test_bounded(self, trending_up: pd.DataFrame) -> None:
        w = T.williams_r(
            trending_up["high"], trending_up["low"], trending_up["close"], 14
        ).dropna()
        assert (w <= 0).all() and (w >= -100).all()


class TestADX:
    def test_bounded(self, trending_up: pd.DataFrame) -> None:
        a = T.adx(trending_up["high"], trending_up["low"], trending_up["close"], 14).dropna()
        assert (a >= 0).all() and (a <= 100).all()


class TestVolatility:
    def test_realized_vol_positive(self) -> None:
        rets = pd.Series(np.random.randn(100) * 0.01)
        v = T.realized_vol(rets, 20).dropna()
        assert (v > 0).all()

    def test_parkinson_positive(self) -> None:
        high = pd.Series(np.linspace(100, 105, 50))
        low = high - 0.5
        v = T.parkinson_vol(high, low, 20).dropna()
        assert (v > 0).all()
