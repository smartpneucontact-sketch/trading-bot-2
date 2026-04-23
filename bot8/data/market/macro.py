"""Macro/market-wide series loader via yfinance.

These feed the regime filter + macro features:
  VIX  — implied volatility (fear index)
  SPY  — S&P 500 ETF (market proxy)
  HYG  — high-yield credit ETF (credit stress)
  TNX  — 10-year Treasury yield
  IRX  — 3-month Treasury yield  (TNX−IRX = yield curve slope)
  GLD  — gold ETF (risk-off proxy)
  DXY  — dollar index  (^DXY on yfinance)
  USO  — oil ETF
  LQD  — investment-grade credit ETF
  SHY  — 1-3yr treasury ETF

yfinance chosen over Alpaca for macro because ^VIX/^TNX/^IRX/^DXY aren't
in Alpaca's stock data coverage. Same library, no extra dependency.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import pandas as pd
from loguru import logger

from bot8.data.db import session

# Series code → yfinance ticker.
# Keeping our internal codes short/ASCII; yfinance uses ^PREFIX for indices.
DEFAULT_SERIES: dict[str, str] = {
    "VIX": "^VIX",
    "SPY": "SPY",
    "HYG": "HYG",
    "TNX": "^TNX",
    "IRX": "^IRX",
    "GLD": "GLD",
    "DXY": "DX-Y.NYB",
    "USO": "USO",
    "LQD": "LQD",
    "SHY": "SHY",
}


def _fetch_series(code: str, yf_ticker: str, start: date, end: date) -> pd.DataFrame:
    """One yfinance download. Returns canonical macro_daily schema."""
    import yfinance as yf

    df = yf.download(
        yf_ticker,
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.min.time()),
        progress=False,
        auto_adjust=False,
        # yfinance can return either multi-index or flat columns depending on version;
        # group_by='ticker' flattens reliably.
        group_by="ticker",
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance returns a MultiIndex if multiple tickers, flat if one.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    df = df.reset_index()
    df = df.rename(columns={
        "Date": "session_date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df["series_code"] = code
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = None

    return df[["series_code", "session_date", "open", "high", "low", "close", "volume"]]


def _insert_series(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    with session() as con:
        con.register("m", df)
        before = con.execute("SELECT COUNT(*) FROM macro_daily").fetchone()[0]
        con.execute(
            """
            INSERT INTO macro_daily (
                series_code, session_date, open, high, low, close, volume
            )
            SELECT m.series_code, m.session_date, m.open, m.high, m.low, m.close, m.volume
            FROM m
            WHERE NOT EXISTS (
                SELECT 1 FROM macro_daily d
                WHERE d.series_code = m.series_code AND d.session_date = m.session_date
            )
            """
        )
        after = con.execute("SELECT COUNT(*) FROM macro_daily").fetchone()[0]
    return after - before


def fetch_macro(
    start: date,
    end: date | None = None,
    series: Iterable[str] | None = None,
) -> dict[str, int]:
    """Fetch macro series into macro_daily. Returns {series_code: rows_inserted}."""
    end = end or date.today()
    codes = list(series) if series else list(DEFAULT_SERIES.keys())

    logger.info("Fetching {} macro series, {} → {}", len(codes), start, end)
    results: dict[str, int] = {}
    for code in codes:
        yf_ticker = DEFAULT_SERIES.get(code)
        if not yf_ticker:
            logger.warning("  unknown series code: {}", code)
            results[code] = -1
            continue
        try:
            df = _fetch_series(code, yf_ticker, start, end)
            results[code] = _insert_series(df)
            logger.info("  {} ({}): {:,} rows", code, yf_ticker, results[code])
        except Exception as e:
            logger.warning("  {} failed: {}", code, e)
            results[code] = -1

    total = sum(n for n in results.values() if n > 0)
    logger.info("Done: {:,} macro rows", total)
    return results
