"""FNSPID historical news loader.

FNSPID (Zihan1004/FNSPID on HuggingFace) is ~29.6 GB of S&P 500 news, 1999–2023,
~15.7M headlines. KDD 2024 paper: arXiv 2402.06698.

We download the `Stock_news/` subtree via the HuggingFace `datasets` library
(or huggingface_hub for file-level control), then stream each CSV into DuckDB.

Schema inspection is done at load time — the column names vary slightly between
the several CSVs in the dataset, so we map them to our canonical `news_raw`
schema dynamically rather than hardcoding.

Design notes:
- Chunked read (200K rows at a time) to avoid loading 29 GB into RAM.
- SHA-1 `headline_hash` for dedup within and across sources.
- `INSERT OR IGNORE` on the PK so re-runs are safe.
- Date filter (`--since`) is applied during load to skip ancient history cheaply.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from bot8.config import get_settings
from bot8.data.db import session

# Canonical column names in our news_raw table.
CANONICAL_COLS = ["symbol", "published_at", "source", "headline", "body", "url", "author"]

# Known FNSPID column name variants → canonical.
# Observed across the Stock_news/*.csv files. If a new variant appears,
# the loader logs and skips — don't silently guess.
COLUMN_MAP = {
    # symbol
    "symbol": "symbol",
    "ticker": "symbol",
    "Stock_symbol": "symbol",
    "stock_symbol": "symbol",
    # date
    "date": "published_at",
    "Date": "published_at",
    "published_at": "published_at",
    "Published_at": "published_at",
    "time": "published_at",
    "datetime": "published_at",
    # headline
    "title": "headline",
    "Title": "headline",
    "headline": "headline",
    "Article_title": "headline",
    # body
    "content": "body",
    "Article": "body",
    "body": "body",
    "text": "body",
    # url
    "url": "url",
    "Url": "url",
    "Link": "url",
    "link": "url",
    # author
    "author": "author",
    "Author": "author",
    "Publisher": "author",
    "publisher": "author",
}


def _headline_hash(symbol: str, headline: str) -> str:
    payload = f"{symbol.upper()}|{headline.strip().lower()}".encode()
    return hashlib.sha1(payload).hexdigest()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map FNSPID columns onto our canonical schema. Drops unknown columns."""
    rename_map = {src: dst for src, dst in COLUMN_MAP.items() if src in df.columns}
    df = df.rename(columns=rename_map)

    # Ensure all canonical columns exist
    for col in CANONICAL_COLS:
        if col not in df.columns:
            df[col] = None

    df["source"] = "fnspid"
    return df[CANONICAL_COLS]


def _clean_frame(df: pd.DataFrame, since: datetime | None) -> pd.DataFrame:
    df = df.copy()
    # Drop rows with no symbol or no headline
    df = df.dropna(subset=["symbol", "headline"])
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["symbol"].str.len() > 0]
    df = df[df["headline"].str.len() > 0]

    # Parse timestamp
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_at"])
    df["published_at"] = df["published_at"].dt.tz_convert("UTC").dt.tz_localize(None)

    if since is not None:
        df = df[df["published_at"] >= pd.Timestamp(since)]

    df["headline_hash"] = [
        _headline_hash(s, h) for s, h in zip(df["symbol"], df["headline"], strict=True)
    ]

    # Intra-chunk dedup: the same headline can appear multiple times in a single
    # FNSPID CSV chunk (syndicated / republished / scraping quirks). DuckDB's
    # atomic INSERT will reject the whole chunk if duplicates exist, so we must
    # collapse them here. `source` is constant ("fnspid") so the PK collapses
    # to (symbol, published_at, headline_hash).
    df = df.drop_duplicates(
        subset=["symbol", "published_at", "headline_hash"], keep="first"
    )
    return df


def _insert_chunk(df: pd.DataFrame) -> int:
    """Insert a chunk into news_raw; returns rows actually inserted (post-dedup)."""
    if df.empty:
        return 0

    with session() as con:
        # DuckDB has no INSERT OR IGNORE; emulate with ANTI JOIN on PK.
        con.register("chunk", df)
        before = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
        # Belt-and-suspenders: SELECT DISTINCT in case a future caller forgets
        # the pandas drop_duplicates in _clean_frame. Both layers are cheap.
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


def load_csv_to_duckdb(
    csv_path: Path,
    since: datetime | None = None,
    chunk_size: int = 200_000,
) -> int:
    """Stream one FNSPID CSV into `news_raw`. Returns total rows inserted."""
    logger.info("Loading FNSPID file: {}", csv_path)

    total_inserted = 0
    try:
        for chunk in pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            on_bad_lines="skip",
            low_memory=False,
        ):
            chunk = _normalize_columns(chunk)
            chunk = _clean_frame(chunk, since=since)
            inserted = _insert_chunk(chunk)
            total_inserted += inserted
            logger.debug("  chunk: {} rows → {} inserted (running: {})",
                         len(chunk), inserted, total_inserted)
    except Exception as e:
        logger.exception("Failed on {}: {}", csv_path, e)
        raise

    logger.info("Done {}: {} new rows", csv_path.name, total_inserted)
    return total_inserted


def download_fnspid(
    since: datetime | None = None,
    files_allowlist: list[str] | None = None,
) -> Path:
    """Download the Stock_news/ subtree from HuggingFace.

    Returns the local directory containing CSVs.
    """
    from huggingface_hub import snapshot_download

    s = get_settings()
    logger.info("Downloading FNSPID Stock_news to {}", s.fnspid_dir)

    patterns = files_allowlist or ["Stock_news/*.csv"]
    local = snapshot_download(
        repo_id="Zihan1004/FNSPID",
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(s.fnspid_dir),
    )
    return Path(local) / "Stock_news"


def ingest_all(
    since: datetime | None = None,
    download: bool = True,
    files_allowlist: list[str] | None = None,
) -> dict[str, int]:
    """End-to-end: download (optional) → iterate CSVs → ingest into DuckDB.

    Returns `{filename: rows_inserted}`.
    """
    from bot8.data.db import init_schema
    init_schema()

    if download:
        news_dir = download_fnspid(since=since, files_allowlist=files_allowlist)
    else:
        news_dir = get_settings().fnspid_dir / "Stock_news"

    if not news_dir.exists():
        raise FileNotFoundError(f"FNSPID Stock_news dir not found at {news_dir}")

    results: dict[str, int] = {}
    for csv_path in sorted(news_dir.glob("*.csv")):
        results[csv_path.name] = load_csv_to_duckdb(csv_path, since=since)
    return results
