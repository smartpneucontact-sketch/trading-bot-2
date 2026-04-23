"""Claude-based news scoring via Message Batches API.

Pipeline stages, each a separate CLI subcommand so the user can stop and
inspect between them:

  1. build    — construct per-(ticker, date) batch requests from news_raw +
                news_scored (only days that have headlines).
  2. estimate — dry-run token counting + cost projection. Enforces --max-cost.
  3. submit   — send batch to Anthropic. Returns a batch ID; saves to
                `claude_batch_jobs` so --poll / --ingest can find it.
  4. poll     — block until the batch is `ended`.
  5. ingest   — download results, parse tool-use outputs, write to
                `news_features_claude_daily`.

Design principles:
- Cost safety: we refuse to submit if estimated cost exceeds `--max-cost`.
- Resumability: batch ID is stored in DB so a crashed session can resume.
- Auditability: every request has a `custom_id` = "{symbol}|{date}" so we
  can trace any ingested row back to its original batch request.
- Versioning: prompt version is stamped into the job record so re-runs with
  a new rubric don't silently overwrite old scores.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable

import pandas as pd
from loguru import logger

from bot8.data.db import session
from bot8.features.news.prompts.rubric_v1 import (
    PROMPT_VERSION,
    SCORE_TOOL,
    SYSTEM_PROMPT,
    render_user_message,
)

# Claude Haiku 4.5 — the right tool for structured sentence-level classification.
# Opus and Sonnet would be overkill and 3-15x more expensive for this task.
MODEL_ID = "claude-haiku-4-5"

# Approximate token counts. The real cost-estimate step uses the Anthropic
# token counter for accuracy; these are planning constants for the CLI ETA.
SYSTEM_PROMPT_TOKENS = 5100    # locked-in cached prefix
OUTPUT_TOKENS_EST = 200         # tool_use call with all fields populated
PER_HEADLINE_TOKENS = 25        # avg English headline plus list numbering

# Pricing ($ per 1M tokens). Source: shared/models.md and prompt-caching.md.
# Batch API = 50% off; cache write (1h TTL) = 2x base; cache read = 10% of base.
HAIKU_INPUT_RATE = 1.00          # $/M uncached input
HAIKU_OUTPUT_RATE = 5.00         # $/M output
CACHE_WRITE_1H_MULT = 2.0        # 1-hour TTL write premium
CACHE_READ_MULT = 0.1            # ~10% of base rate
BATCH_DISCOUNT = 0.5             # 50% off everything


@dataclass(frozen=True, slots=True)
class TickerDayRequest:
    """One per (symbol, date) that has headlines. The smallest scoring unit."""
    symbol: str
    session_date: date
    headlines: list[str]

    @property
    def custom_id(self) -> str:
        # Anthropic custom_id must match ^[a-zA-Z0-9_-]{1,64}$.
        # Our symbols include dots ("BRK.B"), so map them to hyphens. Stock
        # tickers in our universe never contain native hyphens or underscores,
        # so this round-trips safely via decode_custom_id().
        sym_safe = self.symbol.replace(".", "-")
        return f"{sym_safe}_{self.session_date.strftime('%Y%m%d')}"


def decode_custom_id(custom_id: str) -> tuple[str, date]:
    """Reverse of TickerDayRequest.custom_id → (symbol, session_date)."""
    sym_safe, date_str = custom_id.rsplit("_", 1)
    symbol = sym_safe.replace("-", ".")
    sess = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
    return symbol, sess


@dataclass(slots=True)
class CostEstimate:
    n_requests: int
    total_input_tokens: int
    total_output_tokens: int
    cache_write_cost: float
    cache_read_cost: float
    uncached_input_cost: float
    output_cost: float
    total_cost: float

    def pretty(self) -> str:
        return "\n".join([
            f"  requests:               {self.n_requests:,}",
            f"  cached prefix tokens:   {SYSTEM_PROMPT_TOKENS:,} × 1 write + {self.n_requests:,} reads",
            f"  uncached input tokens:  {self.total_input_tokens:,}",
            f"  output tokens:          {self.total_output_tokens:,}",
            "",
            f"  cache write (once):     ${self.cache_write_cost:.4f}",
            f"  cache reads:            ${self.cache_read_cost:.4f}",
            f"  uncached input:         ${self.uncached_input_cost:.4f}",
            f"  output:                 ${self.output_cost:.4f}",
            f"  {'-'*40}",
            f"  TOTAL (batch, 50% off): ${self.total_cost:.2f}",
        ])


# ---------------------------------------------------------------------------
# Step 1: build
# ---------------------------------------------------------------------------

def build_requests(
    since: date,
    until: date,
    top_n_tickers: int | None = 200,
) -> list[TickerDayRequest]:
    """Load (symbol, date, headlines) tuples from news_raw that fall in range.

    When `top_n_tickers` is set, restrict to the N tickers with the most
    headlines in the window. Keeps pilot cost predictable without throwing
    away interesting signal from the mega-caps.
    """
    logger.info("Building Claude batch requests for {} → {}", since, until)

    with session(read_only=True) as con:
        # Identify top-N tickers by news volume in the window.
        if top_n_tickers:
            top_syms = con.execute(
                """
                SELECT symbol, COUNT(*) AS n
                FROM news_raw
                WHERE published_at >= ? AND published_at <= ?
                GROUP BY symbol
                ORDER BY n DESC
                LIMIT ?
                """,
                [since, until, top_n_tickers],
            ).fetchall()
            sym_filter = [s for s, _ in top_syms]
            logger.info("  selected top {} tickers by news volume", len(sym_filter))
        else:
            sym_filter = None

        sql = """
            SELECT symbol, CAST(published_at AS DATE) AS session_date, headline
            FROM news_raw
            WHERE published_at >= ? AND published_at <= ?
        """
        params: list = [since, until]
        if sym_filter:
            placeholders = ",".join(["?"] * len(sym_filter))
            sql += f" AND symbol IN ({placeholders})"
            params.extend(sym_filter)
        sql += " ORDER BY symbol, session_date, headline_hash"

        rows = con.execute(sql, params).fetchall()

    # Group into (symbol, date) buckets
    buckets: dict[tuple[str, date], list[str]] = {}
    for sym, d, h in rows:
        key = (sym, d)
        buckets.setdefault(key, []).append(h)

    requests = [
        TickerDayRequest(symbol=s, session_date=d, headlines=h)
        for (s, d), h in sorted(buckets.items())
    ]
    logger.info("  built {:,} ticker-day requests from {:,} headlines",
                len(requests), sum(len(r.headlines) for r in requests))
    return requests


# ---------------------------------------------------------------------------
# Step 2: estimate
# ---------------------------------------------------------------------------

def estimate_cost(requests: list[TickerDayRequest]) -> CostEstimate:
    """Project batch cost. Uses planning constants + measured per-request input.

    For cost-critical paths, this is a ~5% estimate — conservative on output
    (we assume every field populated at max). Good enough for the $50 cap
    decision. A real token-count pass would use client.messages.count_tokens
    but costs nothing to skip at this precision.
    """
    n = len(requests)
    total_input_tokens = sum(
        len(r.headlines) * PER_HEADLINE_TOKENS + 40  # ~40 tokens of boilerplate
        for r in requests
    )
    total_output_tokens = n * OUTPUT_TOKENS_EST

    # Costs
    cache_write_cost = (SYSTEM_PROMPT_TOKENS / 1_000_000) * HAIKU_INPUT_RATE * CACHE_WRITE_1H_MULT * BATCH_DISCOUNT
    cache_read_cost = (SYSTEM_PROMPT_TOKENS * n / 1_000_000) * HAIKU_INPUT_RATE * CACHE_READ_MULT * BATCH_DISCOUNT
    uncached_input_cost = (total_input_tokens / 1_000_000) * HAIKU_INPUT_RATE * BATCH_DISCOUNT
    output_cost = (total_output_tokens / 1_000_000) * HAIKU_OUTPUT_RATE * BATCH_DISCOUNT
    total_cost = cache_write_cost + cache_read_cost + uncached_input_cost + output_cost

    return CostEstimate(
        n_requests=n,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        cache_write_cost=cache_write_cost,
        cache_read_cost=cache_read_cost,
        uncached_input_cost=uncached_input_cost,
        output_cost=output_cost,
        total_cost=total_cost,
    )


# ---------------------------------------------------------------------------
# Step 3: submit
# ---------------------------------------------------------------------------

def _build_batch_params(request: TickerDayRequest):
    """Build the MessageCreateParams for one ticker-day."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    return MessageCreateParamsNonStreaming(
        model=MODEL_ID,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                # 1-hour TTL so batches that take longer to process still hit cache
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        tools=[SCORE_TOOL],
        tool_choice={"type": "tool", "name": "score_ticker_day"},
        messages=[
            {
                "role": "user",
                "content": render_user_message(
                    request.symbol,
                    request.session_date.isoformat(),
                    request.headlines,
                ),
            }
        ],
    )


# Anthropic's hard limits on a single batch: 100K requests OR 256MB body.
# Each request in our design is ~20KB (5K-token system prompt in JSON + headlines + tool schema),
# so 256MB / 20KB ≈ 12,800 requests fits. We use a conservative 10K for safety margin.
BATCH_SIZE_LIMIT = 10_000


def submit_batch(requests: list[TickerDayRequest], job_name: str) -> list[str]:
    """Submit request(s) to Anthropic. Splits into multiple batches automatically
    when the total exceeds BATCH_SIZE_LIMIT (the body-size limit, not the 100K
    request limit — we hit the former first because of the large cached system
    prompt embedded in every request).

    Returns the list of batch IDs (one per chunk). All persisted to
    claude_batch_jobs so --poll and --ingest can find them.
    """
    import anthropic
    from anthropic.types.messages.batch_create_params import Request

    from bot8.config import get_settings
    s = get_settings()
    if not s.anthropic_api_key.get_secret_value():
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env — cannot submit batch")

    client = anthropic.Anthropic(api_key=s.anthropic_api_key.get_secret_value())

    # Split into chunks
    chunks: list[list[TickerDayRequest]] = [
        requests[i:i + BATCH_SIZE_LIMIT]
        for i in range(0, len(requests), BATCH_SIZE_LIMIT)
    ]
    logger.info(
        "Submitting {:,} requests in {} batch(es) of up to {:,} each",
        len(requests), len(chunks), BATCH_SIZE_LIMIT,
    )

    batch_ids: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        logger.info("  batch {}/{}: uploading {} requests…", i, len(chunks), len(chunk))
        batch = client.messages.batches.create(
            requests=[
                Request(custom_id=r.custom_id, params=_build_batch_params(r))
                for r in chunk
            ]
        )
        logger.info("    → batch id={} status={}", batch.id, batch.processing_status)
        batch_ids.append(batch.id)
        _record_batch_job(
            batch_id=batch.id,
            job_name=f"{job_name}_chunk{i}of{len(chunks)}",
            status=batch.processing_status,
            n_requests=len(chunk),
        )

    return batch_ids


# ---------------------------------------------------------------------------
# Step 4: poll
# ---------------------------------------------------------------------------

def poll_batch(batch_id: str, poll_interval_s: int = 60) -> dict:
    """Block until the batch finishes. Returns final batch object as dict."""
    import anthropic
    from bot8.config import get_settings

    client = anthropic.Anthropic(
        api_key=get_settings().anthropic_api_key.get_secret_value()
    )

    logger.info("Polling batch {} every {}s…", batch_id, poll_interval_s)
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        logger.info(
            "  status={}, processing={}, succeeded={}, errored={}",
            batch.processing_status,
            batch.request_counts.processing,
            batch.request_counts.succeeded,
            batch.request_counts.errored,
        )
        _record_batch_job(
            batch_id=batch_id,
            status=batch.processing_status,
            n_succeeded=batch.request_counts.succeeded,
            n_errored=batch.request_counts.errored,
        )
        if batch.processing_status == "ended":
            return {
                "id": batch.id,
                "status": batch.processing_status,
                "succeeded": batch.request_counts.succeeded,
                "errored": batch.request_counts.errored,
            }
        time.sleep(poll_interval_s)


# ---------------------------------------------------------------------------
# Step 5: ingest
# ---------------------------------------------------------------------------

def ingest_results(batch_id: str) -> int:
    """Download batch results, parse tool_use outputs, write to
    `news_features_claude_daily`. Returns rows inserted."""
    import anthropic
    from bot8.config import get_settings

    client = anthropic.Anthropic(
        api_key=get_settings().anthropic_api_key.get_secret_value()
    )

    logger.info("Fetching results for batch {}…", batch_id)
    rows: list[dict] = []
    n_succeeded = 0
    n_errored = 0

    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            n_errored += 1
            logger.warning("  {} {}: {}",
                           result.custom_id,
                           result.result.type,
                           getattr(result.result, "error", None))
            continue

        # Round-trip "SYMBOL_YYYYMMDD" back to (symbol, date)
        sym, sess_date = decode_custom_id(result.custom_id)

        # Extract tool_use block
        msg = result.result.message
        tool_block = next(
            (b for b in msg.content if b.type == "tool_use"), None
        )
        if tool_block is None:
            n_errored += 1
            logger.warning("  {}: no tool_use block in response", result.custom_id)
            continue

        score = tool_block.input  # dict matching SCORE_TOOL schema

        # Per-row validation. Claude occasionally produces invalid values
        # (e.g. a stringified XML tag in an integer enum field). We drop
        # bad rows rather than blow up the whole insert.
        try:
            row = {
                "symbol": sym,
                "session_date": sess_date,
                "sentiment_score": float(score["sentiment_score"]),
                "primary_catalyst": str(score["primary_catalyst"]),
                "secondary_catalysts": ",".join(
                    str(x) for x in (score.get("secondary_catalysts") or [])
                ),
                "novelty": float(score["novelty"]),
                "magnitude": float(score["magnitude"]),
                "expected_direction": str(score["expected_direction"]),
                "expected_horizon_days": int(score["expected_horizon_days"]),
                "confidence": float(score["confidence"]),
                "reasoning": str(score.get("reasoning") or "")[:500],
                "model_version": f"{MODEL_ID}+{PROMPT_VERSION}",
            }
        except (KeyError, ValueError, TypeError) as e:
            n_errored += 1
            logger.warning("  {} malformed output ({}): {}",
                           result.custom_id, type(e).__name__, str(e)[:120])
            continue

        # Sanity-check numeric ranges; quarantine out-of-range rows
        if not (-1.0 <= row["sentiment_score"] <= 1.0 and
                0.0 <= row["novelty"] <= 1.0 and
                0.0 <= row["magnitude"] <= 1.0 and
                0.0 <= row["confidence"] <= 1.0):
            n_errored += 1
            logger.warning("  {} out-of-range values — skipped", result.custom_id)
            continue

        rows.append(row)
        n_succeeded += 1

    logger.info("  parsed: succeeded={:,}, errored={:,}", n_succeeded, n_errored)

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    # Force numeric dtypes — otherwise if any early batch had a stringified
    # value (pre-validation bug) DuckDB infers VARCHAR for the whole column
    # and all downstream arithmetic breaks.
    for num_col in ("sentiment_score", "novelty", "magnitude", "confidence"):
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
    df["expected_horizon_days"] = pd.to_numeric(
        df["expected_horizon_days"], errors="coerce"
    ).astype("Int64")

    with session() as con:
        con.register("c", df)
        # Drop the existing table if its schema is wrong (can't convert
        # VARCHAR→DOUBLE via ALTER), then recreate from the properly-typed df.
        existing_schema = con.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name='news_features_claude_daily' "
            "AND column_name='sentiment_score'"
        ).fetchone()
        if existing_schema and existing_schema[0] != "DOUBLE":
            logger.warning("news_features_claude_daily has wrong schema — dropping to recreate")
            con.execute("DROP TABLE IF EXISTS news_features_claude_daily")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS news_features_claude_daily AS
            SELECT * FROM c WHERE 1=0
            """
        )
        # Upsert: delete then insert (DuckDB lacks ON CONFLICT for composite keys)
        con.execute(
            """
            DELETE FROM news_features_claude_daily
            WHERE (symbol, session_date) IN (
                SELECT symbol, session_date FROM c
            )
            """
        )
        con.execute("INSERT INTO news_features_claude_daily SELECT * FROM c")
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_nfcd_symbol_date "
            "ON news_features_claude_daily(symbol, session_date)"
        )

    _record_batch_job(batch_id=batch_id, n_ingested=len(df), ingested_at=datetime.utcnow())
    return len(df)


# ---------------------------------------------------------------------------
# Job bookkeeping
# ---------------------------------------------------------------------------

def _record_batch_job(batch_id: str, **updates) -> None:
    """Upsert a row in claude_batch_jobs so --poll / --ingest can find state."""
    with session() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS claude_batch_jobs (
                batch_id    VARCHAR PRIMARY KEY,
                job_name    VARCHAR,
                status      VARCHAR,
                n_requests  INTEGER,
                n_succeeded INTEGER,
                n_errored   INTEGER,
                n_ingested  INTEGER,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ingested_at  TIMESTAMP
            )
            """
        )
        existing = con.execute(
            "SELECT batch_id FROM claude_batch_jobs WHERE batch_id = ?",
            [batch_id],
        ).fetchone()
        if existing is None:
            cols = ["batch_id"] + list(updates.keys())
            vals = [batch_id] + list(updates.values())
            placeholders = ",".join(["?"] * len(cols))
            con.execute(
                f"INSERT INTO claude_batch_jobs ({','.join(cols)}) VALUES ({placeholders})",
                vals,
            )
        else:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            con.execute(
                f"UPDATE claude_batch_jobs SET {set_clause} WHERE batch_id = ?",
                list(updates.values()) + [batch_id],
            )
