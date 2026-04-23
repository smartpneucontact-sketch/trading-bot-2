"""Per-symbol feature engineering.

Takes a single symbol's daily bars DataFrame (sorted by session_date) and
produces all per-symbol features. Caller handles grouping & pasting.

Feature families:
  - returns        — 1/5/10/20/60/252d; log; overnight
  - volatility     — realized, Parkinson, downside, ratios
  - technicals     — RSI, MACD, stochastic, Williams %R, CCI, ADX, ATR
  - bands          — Bollinger (upper/lower ratios, %B, bandwidth)
  - moving avgs    — SMA/EMA ratios, crossovers, slope
  - volume         — ratios, z-score, OBV-normalized, volume-price corr
  - position       — 52w high/low ratios, drawdown, recovery, run-length
  - statistical    — skew, kurtosis, z-score, autocorrelation
  - calendar       — day-of-week, day/week-of-month, quarter/month-end

Output: wide DataFrame keyed by (symbol, session_date) with N feature columns.
NaNs are expected during warmup (first ~252 days) — ML handles them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot8.features.quant import technicals as T

# Feature version. Bump when feature semantics change so downstream caches invalidate.
FEATURE_VERSION = "quant-v1"


def compute_stock_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute all per-symbol features for one symbol's bar series.

    `bars` must contain columns: symbol, session_date, open, high, low, close,
    adj_close, volume. Must be sorted ascending by session_date and have no
    duplicates.
    """
    if bars.empty:
        return bars.copy()

    df = bars.sort_values("session_date").reset_index(drop=True).copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]
    adj_close = df["adj_close"].fillna(df["close"])

    # --- Returns -----------------------------------------------------------
    log_close = np.log(adj_close.replace(0, np.nan))
    log_ret_1d = log_close.diff()

    df["ret_1d"] = adj_close.pct_change(1)
    df["ret_5d"] = adj_close.pct_change(5)
    df["ret_10d"] = adj_close.pct_change(10)
    df["ret_20d"] = adj_close.pct_change(20)
    df["ret_60d"] = adj_close.pct_change(60)
    df["ret_252d"] = adj_close.pct_change(252)
    df["log_ret_1d"] = log_ret_1d
    df["log_ret_5d"] = log_close.diff(5)
    df["log_ret_20d"] = log_close.diff(20)
    df["ret_acceleration"] = df["ret_20d"] - df["ret_60d"]
    df["overnight_return"] = open_ / close.shift(1) - 1

    # --- Volatility --------------------------------------------------------
    df["realized_vol_5d"] = T.realized_vol(log_ret_1d, 5)
    df["realized_vol_20d"] = T.realized_vol(log_ret_1d, 20)
    df["realized_vol_60d"] = T.realized_vol(log_ret_1d, 60)
    df["parkinson_vol_20d"] = T.parkinson_vol(high, low, 20)
    df["downside_vol_20d"] = T.downside_vol(log_ret_1d, 20)
    df["vol_ratio_5_60"] = df["realized_vol_5d"] / df["realized_vol_60d"].replace(0, np.nan)
    df["vol_change_20d"] = df["realized_vol_20d"] / df["realized_vol_20d"].shift(20).replace(0, np.nan)
    df["atr_14"] = T.atr(high, low, close, 14)
    df["atr_pct_14"] = df["atr_14"] / close  # ATR as % of price — cross-sectionally comparable

    # --- Oscillators -------------------------------------------------------
    df["rsi_14"] = T.rsi(close, 14)
    macd_line, macd_sig, macd_hist = T.macd(close)
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_hist
    df["macd_hist_norm"] = macd_hist / close  # normalize by price

    stoch_k, stoch_d = T.stochastic(high, low, close, 14, 3)
    df["stoch_k_14"] = stoch_k
    df["stoch_d_14"] = stoch_d

    df["williams_r_14"] = T.williams_r(high, low, close, 14)
    df["cci_20"] = T.cci(high, low, close, 20)
    df["adx_14"] = T.adx(high, low, close, 14)

    # --- Bollinger bands ---------------------------------------------------
    bb_upper, bb_mid, bb_lower, bb_pctb, bb_width = T.bollinger(close, 20, 2.0)
    df["bb_upper_ratio"] = close / bb_upper
    df["bb_lower_ratio"] = close / bb_lower
    df["bb_pct_b"] = bb_pctb
    df["bb_width"] = bb_width

    # --- Moving averages ---------------------------------------------------
    for n in (10, 20, 50, 200):
        sma = close.rolling(n).mean()
        df[f"sma_{n}_ratio"] = close / sma.replace(0, np.nan)
    for n in (10, 20, 50):
        ema = close.ewm(span=n, adjust=False).mean()
        df[f"ema_{n}_ratio"] = close / ema.replace(0, np.nan)

    df["ma_spread_10_50"] = (
        close.rolling(10).mean() / close.rolling(50).mean().replace(0, np.nan)
    )
    df["golden_cross"] = (
        close.rolling(50).mean() / close.rolling(200).mean().replace(0, np.nan)
    )
    df["sma_20_slope"] = close.rolling(20).mean().pct_change(5)

    # --- Volume ------------------------------------------------------------
    v20 = volume.rolling(20).mean()
    v5 = volume.rolling(5).mean()
    df["volume_ratio_5d"] = volume / v5.replace(0, np.nan)
    df["volume_ratio_20d"] = volume / v20.replace(0, np.nan)
    df["dollar_volume_20d_ratio"] = (
        (close * volume) / (close * volume).rolling(20).mean().replace(0, np.nan)
    )
    df["volume_zscore_20d"] = (volume - v20) / volume.rolling(20).std().replace(0, np.nan)

    obv = T.obv(close, volume)
    df["obv_normalized"] = (obv - obv.rolling(60).mean()) / obv.rolling(60).std().replace(0, np.nan)
    df["volume_price_corr_20d"] = log_ret_1d.rolling(20).corr(volume.pct_change())

    # --- Position / channel ------------------------------------------------
    df["high_52w_ratio"] = close / high.rolling(252).max().replace(0, np.nan)
    df["low_52w_ratio"] = close / low.rolling(252).min().replace(0, np.nan)
    df["drawdown_20d"] = close / close.rolling(20).max().replace(0, np.nan) - 1
    df["drawdown_60d"] = close / close.rolling(60).max().replace(0, np.nan) - 1

    # Range position within 20d (0 = at low, 1 = at high)
    ll20 = low.rolling(20).min()
    hh20 = high.rolling(20).max()
    df["range_position_20d"] = (close - ll20) / (hh20 - ll20).replace(0, np.nan)

    df["consecutive_up_days"] = _run_length(log_ret_1d > 0)
    df["consecutive_down_days"] = _run_length(log_ret_1d < 0)

    # --- Statistical -------------------------------------------------------
    df["return_skew_20d"] = log_ret_1d.rolling(20).skew()
    df["return_skew_60d"] = log_ret_1d.rolling(60).skew()
    df["return_kurt_20d"] = log_ret_1d.rolling(20).kurt()
    df["return_kurt_60d"] = log_ret_1d.rolling(60).kurt()

    df["return_zscore_20d"] = (
        log_ret_1d - log_ret_1d.rolling(20).mean()
    ) / log_ret_1d.rolling(20).std().replace(0, np.nan)
    df["autocorr_lag1_20d"] = log_ret_1d.rolling(20).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )

    # --- Calendar ----------------------------------------------------------
    dt = pd.to_datetime(df["session_date"])
    df["day_of_week"] = dt.dt.dayofweek.astype("float")
    df["day_of_month"] = dt.dt.day.astype("float")
    df["week_of_month"] = ((dt.dt.day - 1) // 7 + 1).astype("float")
    df["is_month_end"] = dt.dt.is_month_end.astype("float")
    df["is_quarter_end"] = dt.dt.is_quarter_end.astype("float")

    return df


def _run_length(condition: pd.Series) -> pd.Series:
    """Current streak length where `condition` is True. Resets to 0 on False."""
    # Classic pandas idiom: group consecutive runs, cumcount within each run.
    grp = (condition != condition.shift()).cumsum()
    streak = condition.groupby(grp).cumcount() + 1
    return streak.where(condition, 0).astype("float")
