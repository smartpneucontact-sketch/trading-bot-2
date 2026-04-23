"""Reusable technical indicators.

All functions take columns as Series and return a Series of the same length.
Vectorized pandas; no per-row Python loops.

Naming: `_N` suffix = lookback window. These are canonical definitions —
verify against any reputable reference if tuning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Momentum / oscillators
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index, Wilder's smoothing (EMA-like).

    Implementation uses the equivalent form:
        RSI = 100 * avg_gain / (avg_gain + avg_loss)
    which avoids the divide-by-zero issue when a series has had zero losses
    (returns 100 as canonically expected) or zero gains (returns 0).
    NaN only during the initial warmup window.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    denom = avg_gain + avg_loss
    rsi_vals = (100 * avg_gain / denom.replace(0, np.nan)).fillna(50)
    # Clip away floating-point overshoot (e.g. 100.00000000000001 when avg_loss≈0).
    return rsi_vals.clip(lower=0, upper=100)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k: int = 14, d: int = 3):
    """Return (%K, %D). %K is raw, %D is SMA(k)."""
    hh = high.rolling(k).max()
    ll = low.rolling(k).min()
    rng = (hh - ll).replace(0, np.nan)
    stoch_k = 100 * (close - ll) / rng
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               n: int = 14) -> pd.Series:
    """Williams %R — inverse of stochastic %K, scaled to [-100, 0]."""
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    rng = (hh - ll).replace(0, np.nan)
    return -100 * (hh - close) / rng


def cci(high: pd.Series, low: pd.Series, close: pd.Series,
        n: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(n).mean()
    mean_dev = tp.rolling(n).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    return (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        n: int = 14) -> pd.Series:
    """Average Directional Index — trend strength, not direction."""
    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = _true_range(high, low, close)
    atr_n = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(
        alpha=1 / n, adjust=False, min_periods=n).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(
        alpha=1 / n, adjust=False, min_periods=n).mean() / atr_n

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


# ---------------------------------------------------------------------------
# Volatility / range
# ---------------------------------------------------------------------------

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        n: int = 14) -> pd.Series:
    """Average True Range, Wilder's smoothing."""
    return _true_range(high, low, close).ewm(
        alpha=1 / n, adjust=False, min_periods=n
    ).mean()


def parkinson_vol(high: pd.Series, low: pd.Series, n: int = 20) -> pd.Series:
    """Parkinson (high-low based) volatility. Annualized."""
    log_hl = np.log(high / low)
    var = (log_hl ** 2).rolling(n).mean() / (4 * np.log(2))
    return np.sqrt(var * 252)


def realized_vol(returns: pd.Series, n: int = 20) -> pd.Series:
    """Annualized realized vol from log returns."""
    return returns.rolling(n).std() * np.sqrt(252)


def downside_vol(returns: pd.Series, n: int = 20) -> pd.Series:
    """Std dev of negative returns only. Annualized."""
    neg = returns.where(returns < 0, 0)
    return neg.rolling(n).std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# Bands / channels
# ---------------------------------------------------------------------------

def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    """Return (upper, middle, lower, pct_b, bandwidth)."""
    middle = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = middle + k * std
    lower = middle - k * std
    rng = (upper - lower).replace(0, np.nan)
    pct_b = (close - lower) / rng
    bandwidth = rng / middle
    return upper, middle, lower, pct_b, bandwidth


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — cumulative volume signed by close direction."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()
