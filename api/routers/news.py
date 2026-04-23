"""Recent scored news — for the "what the bot is seeing" panel."""

from __future__ import annotations

from datetime import date, datetime

from fastapi import APIRouter, Query
from pydantic import BaseModel

from api.db import api_session

router = APIRouter(prefix="/api", tags=["news"])


class ScoredHeadline(BaseModel):
    symbol: str
    published_at: datetime
    headline: str
    sentiment_label: str | None
    sentiment_score: float | None
    catalyst_tags: str | None


@router.get("/news/recent", response_model=list[ScoredHeadline])
def recent_news(
    limit: int = Query(50, ge=1, le=500),
    symbol: str | None = Query(None, description="Filter to one ticker."),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
) -> list[ScoredHeadline]:
    """Most-recent scored headlines. Optionally filter by ticker / confidence."""
    where = ["s.sentiment_conf >= ?"]
    params: list = [min_confidence]
    if symbol:
        where.append("r.symbol = ?")
        params.append(symbol.upper())
    where_sql = " AND ".join(where)
    params.append(limit)

    with api_session() as con:
        rows = con.execute(
            f"""
            SELECT r.symbol, r.published_at, r.headline,
                   s.sentiment_label, s.sentiment_score, s.catalyst_tags
            FROM news_raw r
            JOIN news_scored s USING (symbol, published_at, headline_hash)
            WHERE {where_sql}
            ORDER BY r.published_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    return [
        ScoredHeadline(
            symbol=r[0], published_at=r[1], headline=r[2],
            sentiment_label=r[3], sentiment_score=r[4], catalyst_tags=r[5],
        )
        for r in rows
    ]
