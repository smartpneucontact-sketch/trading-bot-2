"""Backtest comparison endpoint — all meta variants side by side."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.db import api_session

router = APIRouter(prefix="/api", tags=["backtest"])


class BacktestRow(BaseModel):
    score_col: str
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    avg_turnover: float
    n_days: int


@router.get("/backtest/compare", response_model=list[BacktestRow])
def backtest_compare() -> list[BacktestRow]:
    """All rows in backtest_summary, sorted by Sharpe descending."""
    with api_session() as con:
        rows = con.execute(
            """
            SELECT backtest_name, annual_return, annual_vol, sharpe,
                   max_drawdown, hit_rate_daily, avg_turnover, n_days
            FROM backtest_summary
            ORDER BY sharpe DESC
            """
        ).fetchall()

    return [
        BacktestRow(
            score_col=r[0], annual_return=r[1], annual_vol=r[2],
            sharpe=r[3], max_drawdown=r[4], hit_rate=r[5],
            avg_turnover=r[6], n_days=r[7],
        )
        for r in rows
    ]
