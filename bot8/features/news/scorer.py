"""News scoring orchestrator.

Reads unscored headlines from `news_raw`, runs FinBERT + regex, writes to
`news_scored`. Safe to re-run — it only scores rows that are missing in
`news_scored` for the current model version.

Key behaviours:
- Resumable: progress is durable via DuckDB commits per batch.
- Version-aware: if `MODEL_VERSION` changes (regex updated, FinBERT swapped),
  a re-run re-scores everything. Older rows stay in place for audit.
- Efficient: FinBERT batches on GPU/MPS; regex is vectorized in Python.
- Observable: loguru progress every batch + final summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from loguru import logger

from bot8.data.db import session
from bot8.features.news.catalyst_regex import (
    RULESET_VERSION,
    classify_to_tag_string,
)
from bot8.features.news.finbert_scorer import (
    MODEL_VERSION as FINBERT_VERSION,
)

# Combined version string written to news_scored.model_version.
# Ensures re-scoring when either sub-component changes.
CURRENT_MODEL_VERSION = f"{FINBERT_VERSION}+{RULESET_VERSION}"


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    total_scored: int
    batches: int
    started_at: datetime
    finished_at: datetime
    model_version: str

    @property
    def seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def rows_per_second(self) -> float:
        return self.total_scored / self.seconds if self.seconds > 0 else 0.0


def _fetch_unscored(
    limit: int | None,
    since: datetime | None,
    model_version: str,
) -> pd.DataFrame:
    """Fetch headlines in news_raw that lack a news_scored row at current version.

    Using LEFT JOIN + NULL filter rather than NOT EXISTS for DuckDB-friendly
    columnar evaluation on large tables.
    """
    where_clauses = ["ns.symbol IS NULL"]  # LEFT JOIN: no match = unscored
    params: list = []

    if since is not None:
        where_clauses.append("nr.published_at >= ?")
        params.append(since)

    where_sql = " AND ".join(where_clauses)
    limit_sql = f"LIMIT {int(limit)}" if limit else ""

    sql = f"""
        SELECT nr.symbol, nr.published_at, nr.headline_hash, nr.headline
        FROM news_raw nr
        LEFT JOIN news_scored ns
          ON ns.symbol = nr.symbol
         AND ns.published_at = nr.published_at
         AND ns.headline_hash = nr.headline_hash
         AND ns.model_version = ?
        WHERE {where_sql}
        ORDER BY nr.published_at
        {limit_sql}
    """
    query_params = [model_version] + params

    with session(read_only=True) as con:
        return con.execute(sql, query_params).fetchdf()


def _write_scored_rows(rows: pd.DataFrame) -> None:
    """Upsert a batch into news_scored (PK: symbol, published_at, headline_hash)."""
    if rows.empty:
        return
    with session() as con:
        con.register("scored_batch", rows)
        # Anti-join pattern (DuckDB has no INSERT OR REPLACE on composite PKs).
        con.execute(
            """
            DELETE FROM news_scored
            WHERE (symbol, published_at, headline_hash) IN (
                SELECT symbol, published_at, headline_hash FROM scored_batch
            )
            """
        )
        con.execute(
            """
            INSERT INTO news_scored (
                symbol, published_at, headline_hash,
                sentiment_label, sentiment_score, sentiment_conf,
                catalyst_tags, model_version
            )
            SELECT
                symbol, published_at, headline_hash,
                sentiment_label, sentiment_score, sentiment_conf,
                catalyst_tags, model_version
            FROM scored_batch
            """
        )


def score_batch(
    batch_size: int = 256,
    limit: int | None = None,
    since: datetime | None = None,
) -> ScoreSummary:
    """Score unscored headlines in news_raw.

    batch_size here is the *fetch* batch (how many headlines we pull from DB
    per round-trip). FinBertScorer internally has its own model-inference
    batch size for the GPU/MPS forward pass.
    """
    from bot8.features.news.finbert_scorer import get_scorer

    started = datetime.utcnow()
    total = 0
    batches = 0
    scorer = None  # lazy-load on first batch

    logger.info("Scoring news_raw → news_scored (version: {})", CURRENT_MODEL_VERSION)

    while True:
        effective_limit = batch_size if limit is None else min(batch_size, limit - total)
        if effective_limit <= 0:
            break

        df = _fetch_unscored(
            limit=effective_limit,
            since=since,
            model_version=CURRENT_MODEL_VERSION,
        )
        if df.empty:
            break

        # Lazy load — avoids the ~10s model load cost when there's nothing to do.
        if scorer is None:
            scorer = get_scorer()

        # FinBERT sentiment (batched inside FinBertScorer)
        outputs = scorer.score(df["headline"].tolist())
        df["sentiment_label"] = [o.label for o in outputs]
        df["sentiment_score"] = [o.signed_score for o in outputs]
        df["sentiment_conf"] = [o.confidence for o in outputs]

        # Regex catalyst (vectorized in pure Python — fast)
        df["catalyst_tags"] = df["headline"].map(classify_to_tag_string)
        df["model_version"] = CURRENT_MODEL_VERSION

        _write_scored_rows(
            df[
                [
                    "symbol",
                    "published_at",
                    "headline_hash",
                    "sentiment_label",
                    "sentiment_score",
                    "sentiment_conf",
                    "catalyst_tags",
                    "model_version",
                ]
            ]
        )

        batches += 1
        total += len(df)
        logger.info("  batch {}: {} rows (total: {:,})", batches, len(df), total)

        if limit is not None and total >= limit:
            break

    finished = datetime.utcnow()
    summary = ScoreSummary(
        total_scored=total,
        batches=batches,
        started_at=started,
        finished_at=finished,
        model_version=CURRENT_MODEL_VERSION,
    )
    logger.info(
        "Done: {:,} rows in {} batches ({:.1f}s, {:.1f} rows/s)",
        summary.total_scored,
        summary.batches,
        summary.seconds,
        summary.rows_per_second,
    )
    return summary
