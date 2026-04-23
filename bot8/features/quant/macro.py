"""Macro feature engineering — one row per session_date.

Features computed from `macro_daily`:
  - VIX level, 5d change, z-score over 60d
  - SPY returns at multiple horizons + trend vs 200-SMA
  - HYG returns (credit proxy) + HYG/SPY ratio (credit stress)
  - Yield curve slope (TNX − IRX) + change
  - Dollar strength (DXY 20d)
  - Gold momentum + relative to SPY (risk-off)
  - Oil momentum (USO)
  - Composite regime score matching V6's definition

These get broadcast to every symbol row for the same date downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot8.data.db import session


def _load_macro_wide(since: str | None = None) -> pd.DataFrame:
    """Pull macro_daily, pivot to wide (session_date index, series codes as cols).

    Columns present depend on what's in the DB; missing series are dropped silently.
    """
    with session(read_only=True) as con:
        sql = "SELECT series_code, session_date, close FROM macro_daily"
        params: list = []
        if since:
            sql += " WHERE session_date >= ?"
            params.append(since)
        long_df = con.execute(sql, params).fetchdf()

    if long_df.empty:
        return pd.DataFrame()

    wide = long_df.pivot(index="session_date", columns="series_code", values="close")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index().ffill()  # forward-fill holidays across series


def compute_macro_features(since: str | None = None) -> pd.DataFrame:
    """Return a DataFrame indexed by session_date with all macro features."""
    w = _load_macro_wide(since)
    if w.empty:
        return w

    out = pd.DataFrame(index=w.index)

    # VIX --------------------------------------------------------------
    if "VIX" in w:
        out["vix_level"] = w["VIX"]
        out["vix_change_5d"] = w["VIX"].pct_change(5)
        out["vix_change_20d"] = w["VIX"].pct_change(20)
        out["vix_zscore_60d"] = (
            w["VIX"] - w["VIX"].rolling(60).mean()
        ) / w["VIX"].rolling(60).std().replace(0, np.nan)

    # SPY --------------------------------------------------------------
    if "SPY" in w:
        out["spy_ret_5d"] = w["SPY"].pct_change(5)
        out["spy_ret_20d"] = w["SPY"].pct_change(20)
        out["spy_ret_60d"] = w["SPY"].pct_change(60)
        sma200 = w["SPY"].rolling(200).mean()
        out["spy_vs_sma200"] = w["SPY"] / sma200.replace(0, np.nan) - 1

    # HYG (credit) -----------------------------------------------------
    if "HYG" in w:
        out["hyg_ret_20d"] = w["HYG"].pct_change(20)
    if {"HYG", "SPY"}.issubset(w.columns):
        # Credit stress proxy: when HYG underperforms SPY, credit weakens.
        out["hyg_spy_ratio_20d_change"] = (
            (w["HYG"] / w["SPY"]).pct_change(20)
        )

    # Yield curve ------------------------------------------------------
    if {"TNX", "IRX"}.issubset(w.columns):
        slope = w["TNX"] - w["IRX"]
        out["yield_curve_slope"] = slope
        out["yield_curve_change_20d"] = slope.diff(20)

    # Dollar -----------------------------------------------------------
    if "DXY" in w:
        out["dxy_change_20d"] = w["DXY"].pct_change(20)

    # Gold (risk-off) --------------------------------------------------
    if "GLD" in w:
        out["gld_ret_20d"] = w["GLD"].pct_change(20)
    if {"GLD", "SPY"}.issubset(w.columns):
        out["gld_vs_spy_20d"] = (w["GLD"] / w["SPY"]).pct_change(20)

    # Oil --------------------------------------------------------------
    if "USO" in w:
        out["oil_ret_20d"] = w["USO"].pct_change(20)

    # Composite regime score (V6-style). Higher = risk-on.
    # Uses whichever components are available; each term scales to [0, 1].
    # Weights roughly match V6: VIX 30%, SPY trend 30%, SPY momentum 20%, HYG 20%.
    components: list[pd.Series] = []
    weights: list[float] = []
    if "vix_level" in out:
        # Low VIX = risk-on. Map 12→1, 35→0.
        vix_norm = ((35 - out["vix_level"]).clip(0, 23) / 23)
        components.append(vix_norm.fillna(0.5))
        weights.append(0.30)
    if "spy_vs_sma200" in out:
        # SPY above 200-SMA is risk-on. Map to sigmoid-ish.
        trend = 1.0 / (1.0 + np.exp(-20 * out["spy_vs_sma200"]))
        components.append(trend.fillna(0.5))
        weights.append(0.30)
    if "spy_ret_20d" in out:
        mom = 1.0 / (1.0 + np.exp(-20 * out["spy_ret_20d"]))
        components.append(mom.fillna(0.5))
        weights.append(0.20)
    if "hyg_spy_ratio_20d_change" in out:
        cred = 1.0 / (1.0 + np.exp(-20 * out["hyg_spy_ratio_20d_change"]))
        components.append(cred.fillna(0.5))
        weights.append(0.20)

    if components:
        stack = pd.concat(components, axis=1)
        w_arr = np.array(weights) / sum(weights)
        out["regime_score"] = (stack * w_arr).sum(axis=1)
        # V6-style exposure multiplier: maps regime [0, 1] to [0.4, 1.0].
        out["regime_exposure"] = 0.4 + 0.6 * out["regime_score"]

    out.index.name = "session_date"
    out = out.reset_index()
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.date
    return out
