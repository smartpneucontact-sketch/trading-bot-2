"""Risk overlay — regime scaling + drawdown kill switch.

Two risk controls applied AFTER portfolio construction:

1. **Regime scaling**: multiply every weight by the day's `regime_exposure`
   (already computed in `features_quant` via the macro composite: VIX / SPY
   trend / HYG / SPY momentum). Risk-off regimes pull gross exposure down to
   ~40%; risk-on leaves it at 100%. Prevents blowups during crises.

2. **Drawdown kill switch**: if rolling drawdown exceeds the configured
   threshold, set all weights to 0 until equity recovers above the high-water
   mark. Optional — off by default in backtest to measure raw signal quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_regime_scaling(
    weights: pd.DataFrame,
    regime: pd.DataFrame,
    regime_col: str = "regime_exposure",
) -> pd.DataFrame:
    """Multiply each day's weights by that day's regime_exposure.

    `weights`: columns session_date, symbol, weight, ...
    `regime`:  columns session_date, {regime_col} (one row per date).

    Returns a copy of `weights` with `weight` scaled. Days missing regime
    data keep their original weights (assume full exposure).
    """
    out = weights.merge(
        regime[["session_date", regime_col]], on="session_date", how="left"
    )
    out[regime_col] = out[regime_col].fillna(1.0).clip(0.0, 1.0)
    out["weight"] = out["weight"] * out[regime_col]
    return out.drop(columns=[regime_col])


def apply_drawdown_stop(
    daily_returns: pd.Series,
    weights: pd.DataFrame,
    max_drawdown: float = 0.15,
) -> pd.DataFrame:
    """Zero-out weights on days following a drawdown breach, until recovery.

    `daily_returns` must be indexed by session_date (as datetime or date).
    `weights` has session_date column. The stop is evaluated at end of each
    day and applied to the NEXT day's weights (no look-ahead).
    """
    equity = (1 + daily_returns.fillna(0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1

    # Breach days: drawdown below threshold
    breached = dd < -max_drawdown

    # Shift by 1 day so the stop applies to the day AFTER the breach observation
    kill_dates = set(breached[breached].index.shift(1, freq="D"))

    out = weights.copy()
    mask = out["session_date"].isin(kill_dates)
    out.loc[mask, "weight"] = 0.0
    return out
