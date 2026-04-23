"""Trade history endpoint.

Post-live-deployment this reads from `trade_journal`. For now (backtest-only
mode) we synthesize daily "trades" from the weights changing — each
(symbol, date) where the target weight crosses zero or changes sign is a trade.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel

from api.db import api_session

router = APIRouter(prefix="/api", tags=["trades"])


class Trade(BaseModel):
    date: date
    symbol: str
    action: Literal["buy", "sell", "short", "cover"]
    weight_before: float
    weight_after: float
    score: float


@router.get("/trades", response_model=list[Trade])
def recent_trades(
    score_col: str = Query("oof_meta_with_news"),
    limit: int = Query(50, ge=1, le=500),
) -> list[Trade]:
    """Most-recent target-weight changes, synthesized as trade events."""
    with api_session() as con:
        # Weight = score's decile-based sizing, diffed day-over-day.
        rows = con.execute(
            f"""
            WITH ranked AS (
                SELECT
                    session_date, symbol,
                    {score_col} AS score,
                    PERCENT_RANK() OVER (
                        PARTITION BY session_date ORDER BY {score_col}
                    ) AS pct_rank
                FROM quant_oof_preds
                WHERE {score_col} IS NOT NULL
            ),
            weights AS (
                SELECT session_date, symbol, score,
                       CASE WHEN pct_rank >= 0.9 THEN 0.02
                            WHEN pct_rank <= 0.1 THEN -0.02
                            ELSE 0.0 END AS weight
                FROM ranked
            ),
            diffs AS (
                SELECT session_date, symbol, score, weight,
                       LAG(weight) OVER (PARTITION BY symbol ORDER BY session_date) AS prev_weight
                FROM weights
            )
            SELECT session_date, symbol, prev_weight, weight, score
            FROM diffs
            WHERE weight != prev_weight OR (weight = 0 AND prev_weight != 0)
            ORDER BY session_date DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()

    out: list[Trade] = []
    for d, sym, prev, w, score in rows:
        prev = prev or 0.0
        if prev == 0 and w > 0:
            action = "buy"
        elif prev == 0 and w < 0:
            action = "short"
        elif prev > 0 and w <= 0:
            action = "sell"
        elif prev < 0 and w >= 0:
            action = "cover"
        elif w > prev:
            action = "buy"
        else:
            action = "sell"

        out.append(Trade(
            date=d, symbol=sym, action=action,
            weight_before=prev, weight_after=w, score=score,
        ))
    return out
