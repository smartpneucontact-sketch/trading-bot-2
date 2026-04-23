"""Alpaca daily bars loader.

Fetches split/dividend-adjusted daily OHLCV bars from Alpaca Market Data for
every symbol in the universe. Writes to `bars_daily` keyed by (symbol, session_date).

Design:
- Incremental by default: for each symbol, fetch from (last_known_date + 1)
  to `end`. First run pulls everything; subsequent runs only grab missing days.
- Batched per-symbol requests to keep memory flat — Alpaca supports multi-
  symbol queries but we handle symbols individually for clean progress + error
  recovery. Rate limit: 200/min on free tier; we stay comfortably under.
- `adjustment="all"` so splits and dividends are already baked into the prices
  (no need to maintain a split adjustment table).
- Resumable — rerunning is safe (`INSERT OR IGNORE` pattern).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
from loguru import logger

from bot8.data.db import session

# Alpaca free-tier rate limits: 200 req/min. We're nowhere near that, but
# keep a small sleep between symbols to be kind + avoid burst rejection.
_INTER_REQUEST_SLEEP_S = 0.05


def _client():
    """Build an Alpaca historical data client. Lazy import so the `trading`
    extra isn't required for unrelated code paths."""
    from alpaca.data.historical import StockHistoricalDataClient

    from bot8.config import get_settings
    s = get_settings()
    return StockHistoricalDataClient(
        s.alpaca_api_key.get_secret_value(),
        s.alpaca_secret_key.get_secret_value(),
    )


def _last_date_for(symbol: str) -> date | None:
    """Latest session_date we already have for this symbol, or None."""
    with session(read_only=True) as con:
        row = con.execute(
            "SELECT MAX(session_date) FROM bars_daily WHERE symbol = ?", [symbol]
        ).fetchone()
    return row[0] if row and row[0] else None


def _known_range_for(symbol: str) -> tuple[date, date] | None:
    """Return (min_date, max_date) of existing bars for this symbol, or None."""
    with session(read_only=True) as con:
        row = con.execute(
            "SELECT MIN(session_date), MAX(session_date) FROM bars_daily WHERE symbol = ?",
            [symbol],
        ).fetchone()
    if row and row[0] and row[1]:
        return (row[0], row[1])
    return None


def _fetch_symbol(symbol: str, start: date, end: date) -> pd.DataFrame:
    """One Alpaca request. Returns canonical bars schema or empty DF."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.min.time()),
        adjustment=Adjustment.ALL,
        feed="iex",  # free tier; "sip" requires paid sub
    )
    bars = _client().get_stock_bars(req)
    # alpaca-py returns a BarSet; .df is multi-index (symbol, timestamp)
    df = bars.df
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol].drop(columns=["symbol"])
    df = df.rename(columns={"timestamp": "session_date"})
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    df["symbol"] = symbol

    # Normalize the column set for our schema.
    for col in ["open", "high", "low", "close", "volume", "vwap", "trade_count"]:
        if col not in df.columns:
            df[col] = None
    df["adj_close"] = df["close"]  # adjustment=ALL means close is already adjusted

    return df[
        ["symbol", "session_date", "open", "high", "low", "close",
         "adj_close", "volume", "vwap", "trade_count"]
    ]


def _insert_bars(df: pd.DataFrame) -> int:
    """Insert bars into bars_daily; idempotent on PK."""
    if df.empty:
        return 0
    with session() as con:
        con.register("b", df)
        before = con.execute("SELECT COUNT(*) FROM bars_daily").fetchone()[0]
        con.execute(
            """
            INSERT INTO bars_daily (
                symbol, session_date, open, high, low, close,
                adj_close, volume, vwap, trade_count
            )
            SELECT b.symbol, b.session_date, b.open, b.high, b.low, b.close,
                   b.adj_close, b.volume, b.vwap, b.trade_count
            FROM b
            WHERE NOT EXISTS (
                SELECT 1 FROM bars_daily d
                WHERE d.symbol = b.symbol AND d.session_date = b.session_date
            )
            """
        )
        after = con.execute("SELECT COUNT(*) FROM bars_daily").fetchone()[0]
    return after - before


def fetch_bars(
    symbols: Iterable[str],
    start: date,
    end: date | None = None,
    incremental: bool = True,
) -> dict[str, int]:
    """Fetch daily bars for a list of symbols. Returns {symbol: rows_inserted}."""
    import time

    end = end or date.today()
    results: dict[str, int] = {}
    symbols_list = list(symbols)

    logger.info(
        "Fetching bars for {} symbols, {} → {} (incremental={})",
        len(symbols_list),
        start,
        end,
        incremental,
    )

    for i, sym in enumerate(symbols_list, 1):
        # Compute which date ranges still need fetching.
        # Incremental mode must handle BOTH gaps — older-than-existing AND
        # newer-than-existing — so a subsequent backfill with an earlier
        # `--since` picks up the missing historical data. Previous logic only
        # went forward, which silently skipped symbols whose stored range
        # didn't yet cover the requested window.
        gaps: list[tuple[date, date]] = []
        if incremental:
            rng = _known_range_for(sym)
            if rng is None:
                gaps = [(start, end)]
            else:
                min_d, max_d = rng
                if start < min_d:
                    gaps.append((start, min_d - timedelta(days=1)))
                if end > max_d:
                    gaps.append((max_d + timedelta(days=1), end))
        else:
            gaps = [(start, end)]

        if not gaps:
            results[sym] = 0
            continue

        inserted_total = 0
        failed = False
        for gap_start, gap_end in gaps:
            try:
                df = _fetch_symbol(sym, gap_start, gap_end)
                inserted_total += _insert_bars(df)
            except Exception as e:
                logger.warning("  {} [{}-{}] failed: {}", sym, gap_start, gap_end, e)
                failed = True
        results[sym] = inserted_total if not failed else -1

        if i % 25 == 0:
            logger.info("  progress: {}/{} symbols", i, len(symbols_list))
        time.sleep(_INTER_REQUEST_SLEEP_S)

    total = sum(n for n in results.values() if n > 0)
    failed = sum(1 for n in results.values() if n < 0)
    logger.info("Done: {:,} rows inserted across {} symbols ({} failed)",
                total, len(symbols_list), failed)
    return results
