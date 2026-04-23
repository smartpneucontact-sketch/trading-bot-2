"""Live news fetcher — Alpaca News API (primary) + Yahoo Finance (fallback).

FNSPID stops at 2024-01-09. Everything after that needs live feeds. Both
sources write into the same `news_raw` table the backfill uses, so the
FinBERT scoring pipeline runs unchanged.

Design:
- **Alpaca News API is primary**. Free with any Alpaca account, rate limit
  200/min. Returns structured JSON with author + URL + body. Goes back
  roughly 8 years if needed, but we only fetch incremental (since last
  stored timestamp per symbol).
- **Yahoo via yfinance** is fallback / supplement. Free, noisy, shallow
  history, but adds coverage on names Alpaca sometimes misses. We de-dup
  by `headline_hash` so overlap is fine.
- **Incremental by default**: per symbol, we only fetch after the latest
  stored `published_at`. First run pulls `--since` days of backfill.
- **Idempotent writes**: same SHA1 dedup as the FNSPID loader, so repeat
  runs never create duplicates.

Schedule: call `fetch_alpaca_news_incremental()` in the pre-market runner
every day at ~07:00 ET. Takes ~30s for 500 symbols.
"""

from __future__ import annotations

import hashlib
import time
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
from loguru import logger

from bot8.config import get_settings
from bot8.data.db import session


# Alpaca News API: 200 req/min free tier. We batch 10 symbols per request
# and sleep between calls — well under the limit.
ALPACA_NEWS_BATCH_SIZE = 10
INTER_REQUEST_SLEEP_S = 0.3


# ---------------------------------------------------------------------------
# Shared: headline hash for dedup, reused from FNSPID loader
# ---------------------------------------------------------------------------

def _headline_hash(symbol: str, headline: str) -> str:
    payload = f"{symbol.upper()}|{headline.strip().lower()}".encode()
    return hashlib.sha1(payload).hexdigest()


def _insert_rows(df: pd.DataFrame) -> int:
    """Idempotent insert into news_raw — skip rows already present by PK."""
    if df.empty:
        return 0
    with session() as con:
        con.register("chunk", df)
        before = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
        con.execute(
            """
            INSERT INTO news_raw (
                symbol, published_at, source, headline, body, url, author, headline_hash
            )
            SELECT DISTINCT ON (c.symbol, c.published_at, c.source, c.headline_hash)
                   c.symbol, c.published_at, c.source, c.headline, c.body,
                   c.url, c.author, c.headline_hash
            FROM chunk c
            WHERE NOT EXISTS (
                SELECT 1 FROM news_raw n
                WHERE n.symbol = c.symbol
                  AND n.published_at = c.published_at
                  AND n.source = c.source
                  AND n.headline_hash = c.headline_hash
            )
            """
        )
        after = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
    return after - before


def _latest_published_per_symbol(symbols: Iterable[str], source: str) -> dict[str, datetime]:
    """For each symbol, the newest published_at we've already stored for a given source."""
    sym_list = list(symbols)
    if not sym_list:
        return {}
    placeholders = ",".join(["?"] * len(sym_list))
    with session(read_only=True) as con:
        rows = con.execute(
            f"""
            SELECT symbol, MAX(published_at)
            FROM news_raw
            WHERE source = ? AND symbol IN ({placeholders})
            GROUP BY symbol
            """,
            [source, *sym_list],
        ).fetchall()
    return {sym: ts for sym, ts in rows if ts is not None}


# ---------------------------------------------------------------------------
# Alpaca News API — primary source
# ---------------------------------------------------------------------------

def _alpaca_news_client():
    """Lazy import — `trading` extra isn't installed in the read-only API image."""
    from alpaca.data.historical.news import NewsClient
    from bot8.config import get_settings
    s = get_settings()
    return NewsClient(
        s.alpaca_api_key.get_secret_value(),
        s.alpaca_secret_key.get_secret_value(),
    )


def fetch_alpaca_news(
    symbols: list[str],
    start: datetime,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Fetch news from Alpaca for a list of symbols in a time window.

    Returns a DataFrame with the canonical news_raw schema + source='alpaca'.
    """
    from alpaca.data.requests import NewsRequest

    end = end or datetime.utcnow()
    client = _alpaca_news_client()
    rows: list[dict] = []

    for i in range(0, len(symbols), ALPACA_NEWS_BATCH_SIZE):
        batch = symbols[i : i + ALPACA_NEWS_BATCH_SIZE]
        try:
            req = NewsRequest(
                symbols=batch,
                start=start,
                end=end,
                limit=50,  # Alpaca caps at 50 per request
                sort="desc",
            )
            resp = client.get_news(req)
            # Response shape: .news is a list of NewsObject
            articles = resp.news if hasattr(resp, "news") else resp.get("news", [])
            for a in articles:
                # Each article may mention multiple symbols — emit one row per ticker
                article_symbols = [s for s in (a.symbols or []) if s in batch]
                for sym in article_symbols:
                    headline = (a.headline or "").strip()
                    if not headline:
                        continue
                    rows.append({
                        "symbol": sym.upper(),
                        "published_at": a.created_at,
                        "source": "alpaca",
                        "headline": headline,
                        "body": (a.content or a.summary or "").strip() or None,
                        "url": str(a.url) if a.url else None,
                        "author": a.author or None,
                        "headline_hash": _headline_hash(sym, headline),
                    })
        except Exception as e:
            logger.warning("Alpaca news batch {}–{} failed: {}",
                           batch[0] if batch else "?",
                           batch[-1] if batch else "?",
                           e)
        time.sleep(INTER_REQUEST_SLEEP_S)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Normalize timestamp to naive UTC (same as FNSPID loader)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True).dt.tz_localize(None)
    # Intra-batch dedup (same article can appear twice when paginating)
    df = df.drop_duplicates(subset=["symbol", "published_at", "headline_hash"])
    return df


def fetch_alpaca_news_incremental(
    symbols: list[str],
    lookback_days: int = 3,
) -> int:
    """Incremental fetch: per symbol, pick up where we left off.

    `lookback_days` is the backfill window on first run (for new symbols
    that have no stored news yet). Subsequent calls resume from the last
    stored `published_at`. Returns total rows inserted.
    """
    if not symbols:
        return 0

    last_seen = _latest_published_per_symbol(symbols, source="alpaca")
    now = datetime.utcnow()
    default_start = now - timedelta(days=lookback_days)

    # Group symbols by their fetch-start timestamp. Most are "fresh" and can
    # share a common start; a few laggards may need a further-back start.
    by_start: dict[datetime, list[str]] = {}
    for sym in symbols:
        start = last_seen.get(sym, default_start)
        # Add a small safety bump so we don't miss clock-skew edge headlines
        start = start - timedelta(minutes=5)
        by_start.setdefault(start, []).append(sym)

    total = 0
    for start, group in by_start.items():
        logger.info("Alpaca news: {} symbols since {:%Y-%m-%d %H:%M}", len(group), start)
        df = fetch_alpaca_news(group, start=start, end=now)
        if df.empty:
            continue
        n = _insert_rows(df)
        total += n
        logger.info("  inserted {} new rows", n)

    logger.info("Alpaca news: {} total new rows across {} symbols", total, len(symbols))
    return total


# ---------------------------------------------------------------------------
# Yahoo Finance — fallback / supplement
# ---------------------------------------------------------------------------

def fetch_yahoo_news(symbol: str) -> pd.DataFrame:
    """Fetch recent news for one symbol from yfinance. Shallow history."""
    import yfinance as yf
    try:
        news = yf.Ticker(symbol).news  # list of dicts
    except Exception as e:
        logger.warning("Yahoo news for {}: {}", symbol, e)
        return pd.DataFrame()

    rows: list[dict] = []
    for item in news or []:
        # yfinance v0.2+ wraps data in a "content" subdict; handle both.
        content = item.get("content", item)
        headline = (content.get("title") or "").strip()
        if not headline:
            continue
        pub = content.get("pubDate") or content.get("providerPublishTime")
        if isinstance(pub, (int, float)):
            published_at = datetime.utcfromtimestamp(pub)
        else:
            try:
                published_at = pd.Timestamp(pub).tz_convert("UTC").tz_localize(None).to_pydatetime()
            except Exception:
                continue
        url = (content.get("canonicalUrl") or {}).get("url") or content.get("link")
        rows.append({
            "symbol": symbol.upper(),
            "published_at": published_at,
            "source": "yahoo",
            "headline": headline,
            "body": (content.get("summary") or "").strip() or None,
            "url": url,
            "author": (content.get("provider") or {}).get("displayName") or None,
            "headline_hash": _headline_hash(symbol, headline),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["symbol", "published_at", "headline_hash"])


def fetch_yahoo_news_batch(symbols: list[str]) -> int:
    """Fetch latest Yahoo news for many symbols, idempotent insert."""
    total = 0
    for sym in symbols:
        df = fetch_yahoo_news(sym)
        if df.empty:
            continue
        total += _insert_rows(df)
        time.sleep(0.1)  # yfinance is rate-lax but don't hammer it
    logger.info("Yahoo news: {} new rows across {} symbols", total, len(symbols))
    return total
