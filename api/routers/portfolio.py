"""Portfolio + positions endpoints.

Reads from `quant_oof_preds` (the authoritative target-weights table) and
`bars_daily` (for current prices). In live-trading mode this will also read
from a new `live_positions` table (filled by the Alpaca execution adapter
on every fill) — for now we synthesize a "backtest portfolio" view from the
OOF predictions so the UI has something real to show.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel

from api.db import api_session

router = APIRouter(prefix="/api", tags=["portfolio"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class Position(BaseModel):
    symbol: str
    weight: float
    side: Literal["long", "short"]
    score: float


class PortfolioSnapshot(BaseModel):
    as_of_date: date
    n_longs: int
    n_shorts: int
    gross_exposure: float
    net_exposure: float
    positions: list[Position]


class EquityPoint(BaseModel):
    date: date
    equity: float
    drawdown: float
    gross_pnl: float
    net_pnl: float


class MetricsSummary(BaseModel):
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    avg_turnover: float
    n_days: int
    score_col: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/portfolio", response_model=PortfolioSnapshot)
def current_portfolio(
    score_col: str = Query(
        "oof_meta_with_news",
        description="Which OOF meta to read weights from.",
    ),
) -> PortfolioSnapshot:
    """Current target portfolio — last available date in the backtest."""
    with api_session() as con:
        # Latest date present in the backtest
        as_of = con.execute(
            "SELECT MAX(session_date) FROM quant_oof_preds"
        ).fetchone()[0]
        if as_of is None:
            return PortfolioSnapshot(
                as_of_date=date.today(), n_longs=0, n_shorts=0,
                gross_exposure=0.0, net_exposure=0.0, positions=[]
            )

        # Pick top/bottom decile of score as positions (matches backtest logic)
        rows = con.execute(
            f"""
            WITH ranked AS (
                SELECT symbol, {score_col} AS score,
                       PERCENT_RANK() OVER (ORDER BY {score_col}) AS pct_rank
                FROM quant_oof_preds
                WHERE session_date = ? AND {score_col} IS NOT NULL
            )
            SELECT symbol, score,
                   CASE WHEN pct_rank >= 0.9 THEN 'long'
                        WHEN pct_rank <= 0.1 THEN 'short'
                        END AS side
            FROM ranked
            WHERE pct_rank >= 0.9 OR pct_rank <= 0.1
            ORDER BY score DESC
            """,
            [as_of],
        ).fetchall()

    positions: list[Position] = []
    n_longs = sum(1 for r in rows if r[2] == "long")
    n_shorts = sum(1 for r in rows if r[2] == "short")
    long_weight = 1.0 / n_longs if n_longs else 0.0
    short_weight = 1.0 / n_shorts if n_shorts else 0.0

    for sym, score, side in rows:
        w = long_weight if side == "long" else -short_weight
        positions.append(Position(symbol=sym, weight=w, side=side, score=score))

    gross = sum(abs(p.weight) for p in positions)
    net = sum(p.weight for p in positions)

    return PortfolioSnapshot(
        as_of_date=as_of,
        n_longs=n_longs,
        n_shorts=n_shorts,
        gross_exposure=gross,
        net_exposure=net,
        positions=positions,
    )


@router.get("/equity-curve", response_model=list[EquityPoint])
def equity_curve(
    score_col: str = Query("oof_meta_with_news"),
    days: int = Query(365, ge=1, le=5000),
) -> list[EquityPoint]:
    """Equity curve for a given backtest score. Reads `backtest_daily`."""
    with api_session() as con:
        rows = con.execute(
            """
            SELECT session_date, equity, drawdown, gross_pnl, net_pnl
            FROM backtest_daily
            WHERE backtest_name = ?
              AND session_date >= (
                  SELECT MAX(session_date) - INTERVAL (CAST(? AS INT)) DAY
                  FROM backtest_daily WHERE backtest_name = ?
              )
            ORDER BY session_date
            """,
            [score_col, days, score_col],
        ).fetchall()

    return [
        EquityPoint(
            date=r[0], equity=r[1], drawdown=r[2], gross_pnl=r[3], net_pnl=r[4]
        )
        for r in rows
    ]


@router.get("/metrics", response_model=MetricsSummary)
def metrics(
    score_col: str = Query("oof_meta_with_news"),
) -> MetricsSummary:
    """Aggregate backtest metrics for a given score."""
    with api_session() as con:
        row = con.execute(
            """
            SELECT annual_return, annual_vol, sharpe, max_drawdown,
                   hit_rate_daily, avg_turnover, n_days
            FROM backtest_summary
            WHERE backtest_name = ?
            """,
            [score_col],
        ).fetchone()

    if row is None:
        return MetricsSummary(
            annual_return=0, annual_vol=0, sharpe=0, max_drawdown=0,
            hit_rate=0, avg_turnover=0, n_days=0, score_col=score_col,
        )

    return MetricsSummary(
        annual_return=row[0], annual_vol=row[1], sharpe=row[2],
        max_drawdown=row[3], hit_rate=row[4], avg_turnover=row[5],
        n_days=row[6], score_col=score_col,
    )
