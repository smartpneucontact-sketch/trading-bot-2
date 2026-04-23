"""Cross-sectional features — ranks within universe and within sector per day.

Cross-sectional rank is *the* key ingredient for a long/short strategy. Raw
feature values mean little; what matters is where each stock ranks *today*
against the tradable universe. Converting to per-day rank:

  - normalizes away regime shifts (a high RSI in a calm market ≠ high in volatile)
  - puts every feature on the same [0, 1] scale so tree models don't need scaling
  - is what a dollar-neutral long/short portfolio actually trades on

We also compute sector-relative ranks so the model can separate "this name is
leading its sector" from "this name's sector is leading the market."
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def rank_cross_sectional(
    wide: pd.DataFrame,
    columns: Iterable[str],
    suffix: str = "_rank",
) -> pd.DataFrame:
    """For each date (groupby session_date), compute rank(0,1) of each feature
    across symbols. Handles NaN: ranks within the non-NaN set.

    `wide` must have columns: session_date, symbol, plus the feature cols.
    Returns a new frame with the original PK + new `{col}{suffix}` columns.
    """
    cols = [c for c in columns if c in wide.columns]
    if not cols:
        return pd.DataFrame({"session_date": wide["session_date"], "symbol": wide["symbol"]})

    grouped = wide.groupby("session_date", sort=False)

    out = pd.DataFrame(
        {"session_date": wide["session_date"].values, "symbol": wide["symbol"].values}
    )
    for c in cols:
        out[c + suffix] = grouped[c].rank(pct=True, method="average").values
    return out


def rank_within_sector(
    wide: pd.DataFrame,
    columns: Iterable[str],
    sector_col: str = "sector",
    suffix: str = "_sector_rank",
) -> pd.DataFrame:
    """Rank per (date, sector). Same signature as rank_cross_sectional."""
    cols = [c for c in columns if c in wide.columns]
    if not cols or sector_col not in wide.columns:
        return pd.DataFrame({"session_date": wide["session_date"], "symbol": wide["symbol"]})

    grouped = wide.groupby(["session_date", sector_col], sort=False)

    out = pd.DataFrame(
        {"session_date": wide["session_date"].values, "symbol": wide["symbol"].values}
    )
    for c in cols:
        out[c + suffix] = grouped[c].rank(pct=True, method="average").values
    return out


# Features that benefit most from cross-sectional ranking. Momentum / value /
# volatility / volume all have heavy regime-dependent shifts in raw level.
# These are the canonical ones to rank; the rest stay as raw.
DEFAULT_CROSS_RANK_COLS: tuple[str, ...] = (
    "ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_252d",
    "realized_vol_20d", "realized_vol_60d",
    "rsi_14", "macd_hist_norm",
    "volume_ratio_20d", "volume_zscore_20d",
    "drawdown_60d", "high_52w_ratio",
    "return_skew_20d",
)
