"""Build a minimal 'dashboard' DuckDB containing only the tables the
FastAPI endpoints read — suitable for uploading to Railway's volume.

Full bot8.duckdb = 9.6 GB (news + bars + macro + 118-col features).
Dashboard subset  ≈ 200-500 MB.

Drops:
  - bars_daily              (OHLCV not used by any endpoint)
  - macro_daily             (used only at feature-build time)
  - features_quant          (meta-learner input, not read at runtime)
  - labels_quant            (same)
  - news_features_daily     (aggregated; could add back if needed)
  - news_features_claude_daily
  - all news_raw older than 30 days

Keeps:
  - backtest_summary        (/api/metrics, /api/backtest/compare)
  - backtest_daily          (/api/equity-curve)
  - quant_oof_preds         (/api/portfolio, /api/trades)
  - universe                (sector lookups)
  - news_raw (last 30d)     (/api/news/recent)
  - news_scored (last 30d)  (matches filtered news_raw)
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
from loguru import logger

SRC = Path("data/db/bot8.duckdb")
DST = Path("data/db/bot8-dashboard.duckdb")

NEWS_LOOKBACK_DAYS = 30  # days of news headlines to keep in the dashboard DB
NEWS_LOOKBACK_MULTIPLIER = 1  # was 10 — FNSPID ends 2024-01 so "last 30d" still has content from end of history


def main() -> None:
    if not SRC.exists():
        logger.error("Source DB not found at {}", SRC)
        sys.exit(1)

    if DST.exists():
        DST.unlink()
        logger.info("Removed existing {}", DST)

    # Connect source (read-only) and destination
    src = duckdb.connect(database=str(SRC), read_only=True)
    dst = duckdb.connect(database=str(DST), read_only=False)

    # Dashboard-critical tables — copy whole
    FULL_COPY = [
        "backtest_summary",
        "backtest_daily",
        "quant_oof_preds",
        "universe",
    ]
    for tbl in FULL_COPY:
        exists = src.execute(
            f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{tbl}'"
        ).fetchone()[0]
        if not exists:
            logger.warning("  {} not in source — skipping", tbl)
            continue
        df = src.execute(f"SELECT * FROM {tbl}").fetchdf()
        dst.register("tmp", df)
        dst.execute(f"CREATE TABLE {tbl} AS SELECT * FROM tmp")
        dst.unregister("tmp")
        logger.info("  copied {}: {:,} rows", tbl, len(df))

    # News tables — last N days only
    # Anchor the lookback on the latest date in news_raw (FNSPID ends 2024-01),
    # not today. Otherwise "last 30 days from today" is empty for historical data.
    latest = src.execute("SELECT MAX(published_at) FROM news_raw").fetchone()[0]
    if latest:
        effective_cutoff = (latest - timedelta(days=NEWS_LOOKBACK_DAYS)).date()
    else:
        effective_cutoff = date.today() - timedelta(days=NEWS_LOOKBACK_DAYS)
    logger.info("  news cutoff: {}", effective_cutoff)

    news_raw = src.execute(
        "SELECT * FROM news_raw WHERE published_at >= ?", [effective_cutoff]
    ).fetchdf()
    dst.register("tmp", news_raw)
    dst.execute("CREATE TABLE news_raw AS SELECT * FROM tmp")
    dst.unregister("tmp")
    logger.info("  copied news_raw (filtered): {:,} rows", len(news_raw))

    # news_scored keyed by same PK → filter to match
    news_scored = src.execute(
        """
        SELECT s.*
        FROM news_scored s
        INNER JOIN news_raw r
          ON r.symbol = s.symbol
         AND r.published_at = s.published_at
         AND r.headline_hash = s.headline_hash
        WHERE r.published_at >= ?
        """,
        [effective_cutoff],
    ).fetchdf()
    dst.register("tmp", news_scored)
    dst.execute("CREATE TABLE news_scored AS SELECT * FROM tmp")
    dst.unregister("tmp")
    logger.info("  copied news_scored (filtered): {:,} rows", len(news_scored))

    # Add basic indexes so queries stay fast
    dst.execute("CREATE INDEX idx_qoof_symbol_date ON quant_oof_preds(symbol, session_date)")
    dst.execute("CREATE INDEX idx_nr_date ON news_raw(published_at)")
    dst.execute("CREATE INDEX idx_ns_pk ON news_scored(symbol, published_at, headline_hash)")

    src.close()
    dst.close()

    size_mb = DST.stat().st_size / (1024 * 1024)
    logger.info("✓ Wrote {} — {:.1f} MB", DST, size_mb)


if __name__ == "__main__":
    main()
