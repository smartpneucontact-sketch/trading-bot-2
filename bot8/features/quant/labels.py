"""Label construction — 1-day forward returns, raw and cross-sectionally standardized.

For a long/short strategy, the model should predict the *ranking* of next-day
returns across the cross-section, not the raw magnitude (which is dominated
by market-wide moves). Both forms are stored:

  - fwd_return_1d:       raw next-day return (close[t+1] / close[t] − 1)
  - fwd_return_1d_demean: next-day return minus the cross-sectional mean
                           (market-neutral target)
  - fwd_return_1d_zscore: z-scored within the day's cross-section
  - fwd_rank_1d:         rank in [0, 1] across the day's cross-section

The last two are what the model should actually train on. `fwd_return_1d` is
kept for backtesting P&L and evaluation.

Labels are NaN for the last day of data (can't look forward from the end).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from bot8.data.db import session


def compute_labels(since: date | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns:
       symbol, session_date, fwd_return_1d, fwd_return_1d_demean,
       fwd_return_1d_zscore, fwd_rank_1d.
    """
    with session(read_only=True) as con:
        sql = """
            SELECT symbol, session_date, adj_close
            FROM bars_daily
        """
        params: list = []
        if since:
            sql += " WHERE session_date >= ?"
            params.append(since)
        sql += " ORDER BY symbol, session_date"
        df = con.execute(sql, params).fetchdf()

    if df.empty:
        return df

    # Next-day return per symbol
    df = df.sort_values(["symbol", "session_date"])
    df["fwd_return_1d"] = df.groupby("symbol")["adj_close"].pct_change(1).shift(-1)

    # Cross-sectional stats per date
    grouped = df.groupby("session_date")["fwd_return_1d"]
    cs_mean = grouped.transform("mean")
    cs_std = grouped.transform("std")

    df["fwd_return_1d_demean"] = df["fwd_return_1d"] - cs_mean
    df["fwd_return_1d_zscore"] = (df["fwd_return_1d"] - cs_mean) / cs_std.replace(0, np.nan)
    df["fwd_rank_1d"] = grouped.rank(pct=True, method="average")

    return df[[
        "symbol", "session_date",
        "fwd_return_1d", "fwd_return_1d_demean",
        "fwd_return_1d_zscore", "fwd_rank_1d",
    ]]
