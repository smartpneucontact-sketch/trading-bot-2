"""Universe loader — S&P 500 + Nasdaq 100 from Wikipedia, enriched with
Alpaca asset metadata (shortable, fractionable, tradable).

Design:
- Wikipedia tables are the canonical free source for index constituents.
  Parsed with pandas.read_html — fragile to markup changes, but the only
  realistic free option. If Wikipedia re-skins, we log and fail loudly.
- Alpaca `GET /v2/assets` is the source of truth for tradability + shortable.
  We left-join index constituents onto the Alpaca asset table so the
  universe is always a subset of what we can actually trade.
- The `universe` table is `as_of_date`-keyed so we preserve historical
  membership (survivorship-bias-free). Each refresh writes a new as-of
  snapshot; we don't overwrite old rows.

GICS sector:
- Wikipedia's S&P 500 table has GICS Sector + GICS Sub-Industry columns.
  We use those for sector/industry.
- Nasdaq 100's table has Sector but not GICS; we keep it in the same column.
"""

from __future__ import annotations

import re
from datetime import date
from io import StringIO
from typing import Literal

import pandas as pd
import requests
from loguru import logger

from bot8.data.db import session

# Wikipedia columns often carry footnote refs like "GICS Sector[5]" or
# "ICB Industry[14]". Strip them to canonical column names before matching.
_FOOTNOTE_RE = re.compile(r"\[[^\]]*\]")


def _clean_col(c: str) -> str:
    return _FOOTNOTE_RE.sub("", str(c)).strip()


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_col(c) for c in df.columns]
    return df

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NDX100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

IndexName = Literal["SP500", "NDX100"]

# Wikipedia blocks pandas' default urllib User-Agent with HTTP 403. Use
# requests with a real UA and hand the HTML text to pandas.read_html.
_UA = "Mozilla/5.0 (compatible; bot8/0.1; +https://github.com/local/bot8)"


def _read_html(url: str) -> list[pd.DataFrame]:
    """Fetch URL with a real User-Agent, return list of parsed tables."""
    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=30)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))


def fetch_sp500() -> pd.DataFrame:
    """Return DataFrame with columns: symbol, sector, industry."""
    logger.info("Fetching S&P 500 constituents from Wikipedia")
    tables = _read_html(SP500_URL)
    # First table is the constituents table (columns: Symbol, Security, GICS Sector, GICS Sub-Industry, ...)
    df = _clean_columns(tables[0])
    required = ["Symbol", "GICS Sector", "GICS Sub-Industry"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"S&P 500 Wikipedia table schema changed; missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    df = df[required].copy()
    df.columns = ["symbol", "sector", "industry"]
    # Some tickers are listed with dots (BRK.B) — Alpaca uses dashes (BRK.B → BRK.B; yfinance uses BRK-B).
    # We keep the dotted form as canonical; execution adapters handle the mapping.
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    logger.info("  S&P 500: {} constituents", len(df))
    return df


# Accepted column name aliases (canonical after _clean_col strips footnotes).
# Nasdaq-100 uses ICB classification; S&P 500 uses GICS; some tables just say
# "Sector"/"Industry". Supporting all three keeps the loader resilient to
# Wikipedia re-classifying things.
_SYMBOL_ALIASES = ("Ticker", "Symbol")
_SECTOR_ALIASES = ("GICS Sector", "ICB Industry", "Sector")
_INDUSTRY_ALIASES = ("GICS Sub-Industry", "ICB Subsector", "Sub-Industry", "Industry")


def _pick_column(cols: set[str], aliases: tuple[str, ...]) -> str | None:
    for a in aliases:
        if a in cols:
            return a
    return None


def fetch_ndx100() -> pd.DataFrame:
    """Return DataFrame with columns: symbol, sector, industry."""
    logger.info("Fetching Nasdaq 100 constituents from Wikipedia")
    tables = _read_html(NDX100_URL)

    candidates: list[pd.DataFrame] = []
    for raw in tables:
        t = _clean_columns(raw)
        cols = set(t.columns)
        if _pick_column(cols, _SYMBOL_ALIASES) and _pick_column(cols, _SECTOR_ALIASES):
            candidates.append(t)
    if not candidates:
        raise RuntimeError(
            "Nasdaq 100 Wikipedia page: no table matches the expected shape. "
            f"Tables available: {[list(_clean_columns(t).columns)[:4] for t in tables]}"
        )

    # Take the biggest matching table (most likely the constituents list).
    df = max(candidates, key=len).copy()
    cols = set(df.columns)

    symbol_col = _pick_column(cols, _SYMBOL_ALIASES)
    sector_col = _pick_column(cols, _SECTOR_ALIASES)
    industry_col = _pick_column(cols, _INDUSTRY_ALIASES) or sector_col

    df = df[[symbol_col, sector_col, industry_col]].copy()
    df.columns = ["symbol", "sector", "industry"]
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    logger.info("  Nasdaq 100: {} constituents", len(df))
    return df


def fetch_alpaca_assets() -> pd.DataFrame:
    """Return DataFrame with columns: symbol, is_shortable, is_fractionable,
    is_tradable. Uses the Alpaca trading API; requires keys in .env.
    """
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetStatus

    from bot8.config import get_settings
    s = get_settings()
    logger.info("Fetching tradable assets from Alpaca")

    client = TradingClient(
        s.alpaca_api_key.get_secret_value(),
        s.alpaca_secret_key.get_secret_value(),
        paper=True,
    )
    assets = client.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE))

    rows = [
        {
            "symbol": a.symbol.upper(),
            "is_shortable": bool(a.shortable),
            "is_fractionable": bool(a.fractionable),
            "is_tradable": bool(a.tradable),
        }
        for a in assets
    ]
    df = pd.DataFrame(rows)
    logger.info("  Alpaca: {} active assets ({} shortable)", len(df), df["is_shortable"].sum())
    return df


def build_universe(as_of: date | None = None) -> pd.DataFrame:
    """Merge Wikipedia + Alpaca into the `universe` schema. Returns the frame
    that will be written to DuckDB (not yet inserted)."""
    as_of = as_of or date.today()

    sp500 = fetch_sp500().assign(index_membership="SP500")
    ndx100 = fetch_ndx100().assign(index_membership="NDX100")
    wiki = pd.concat([sp500, ndx100], ignore_index=True)

    alpaca = fetch_alpaca_assets()

    merged = wiki.merge(alpaca, on="symbol", how="left")
    # If a Wikipedia symbol isn't in Alpaca's active asset list, default to not tradable.
    for col in ["is_shortable", "is_fractionable", "is_tradable"]:
        merged[col] = merged[col].fillna(False).astype(bool)

    merged["as_of_date"] = pd.Timestamp(as_of).date()
    return merged[
        [
            "symbol",
            "as_of_date",
            "index_membership",
            "sector",
            "industry",
            "is_shortable",
            "is_fractionable",
            "is_tradable",
        ]
    ]


def refresh_universe(as_of: date | None = None) -> int:
    """Write a fresh as-of snapshot into the `universe` table.

    Idempotent per (symbol, as_of_date, index_membership) — re-running on the
    same day overwrites that day's snapshot.
    """
    df = build_universe(as_of=as_of)

    with session() as con:
        con.register("u", df)
        con.execute(
            """
            DELETE FROM universe
            WHERE (symbol, as_of_date, index_membership) IN (
                SELECT symbol, as_of_date, index_membership FROM u
            )
            """
        )
        con.execute(
            """
            INSERT INTO universe (
                symbol, as_of_date, index_membership, sector, industry,
                is_shortable, is_fractionable, is_tradable
            )
            SELECT symbol, as_of_date, index_membership, sector, industry,
                   is_shortable, is_fractionable, is_tradable
            FROM u
            """
        )
    logger.info("universe snapshot written: {} rows ({})", len(df), df["as_of_date"].iloc[0])
    return len(df)


def current_universe(
    index: IndexName | None = None,
    require_shortable: bool = False,
    require_fractionable: bool = False,
    require_tradable: bool = True,
) -> list[str]:
    """Return the list of symbols in the most recent as-of snapshot, filtered.

    Used downstream to constrain data fetches and backtest universe.
    """
    where = ["as_of_date = (SELECT MAX(as_of_date) FROM universe)"]
    if index:
        where.append(f"index_membership = '{index}'")
    if require_shortable:
        where.append("is_shortable = TRUE")
    if require_fractionable:
        where.append("is_fractionable = TRUE")
    if require_tradable:
        where.append("is_tradable = TRUE")
    where_sql = " AND ".join(where)

    with session(read_only=True) as con:
        rows = con.execute(
            f"SELECT DISTINCT symbol FROM universe WHERE {where_sql} ORDER BY symbol"
        ).fetchall()
    return [r[0] for r in rows]
