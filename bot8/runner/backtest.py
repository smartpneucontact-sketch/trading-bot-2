"""Daily-rebalance long/short backtest.

Pipeline:
  1. Load OOF meta predictions + labels + universe metadata (sector, shortable).
  2. Build daily target weights from predictions (portfolio.longshort).
  3. Apply regime scaling from features_quant.regime_exposure.
  4. Simulate day-by-day:
       daily_pnl_t = Σ_i (weight_i,t × fwd_return_1d_t) − costs_t
       costs_t    = slippage_bps × turnover_t
  5. Build equity curve, Sharpe, drawdown, turnover, hit-rate.

No look-ahead: `weight` at date t is built from info available at t close,
and realized against `fwd_return_1d(t)` which is the close-to-close return
from t to t+1. Purged walk-forward OOF guarantees predictions are out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from bot8.data.db import session
from bot8.portfolio.longshort import PortfolioConfig, build_daily_weights
from bot8.portfolio.risk import apply_regime_scaling


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    slippage_bps: float = 5.0         # one-way, applied on turnover
    borrow_bps_annual: float = 25.0   # annual short borrow cost (per day = /252)
    apply_regime: bool = True
    portfolio: PortfolioConfig = PortfolioConfig()


@dataclass
class BacktestResult:
    daily: pd.DataFrame               # per-date metrics (equity, P&L, gross/net, turnover)
    summary: dict                     # aggregate stats
    weights: pd.DataFrame             # all target weights (for auditing)


def _load_backtest_inputs(score_col: str, since: date | None) -> pd.DataFrame:
    """Build the day-by-symbol table the portfolio builder expects:
    session_date, symbol, score, fwd_return_1d, sector, is_shortable, regime_exposure."""
    where = ""
    params: list = []
    if since:
        where = "WHERE q.session_date >= ?"
        params = [since]

    sql = f"""
        SELECT
            q.session_date,
            q.symbol,
            q.{score_col} AS score,
            q.fwd_return_1d,
            f.regime_exposure,
            f.sector,
            u.is_shortable
        FROM quant_oof_preds q
        JOIN features_quant f USING (symbol, session_date)
        LEFT JOIN (
            SELECT symbol, BOOL_OR(is_shortable) AS is_shortable
            FROM universe
            WHERE as_of_date = (SELECT MAX(as_of_date) FROM universe)
            GROUP BY symbol
        ) u USING (symbol)
        {where}
    """
    with session(read_only=True) as con:
        df = con.execute(sql, params).fetchdf()
    df["is_shortable"] = df["is_shortable"].fillna(False).astype(bool)
    return df


def _simulate(
    weights: pd.DataFrame,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """Per-day P&L, turnover, costs → daily frame."""
    # Long format: (session_date, symbol, weight, fwd_return_1d).
    # Pivot symbol→columns, then vectorize.
    w = weights.pivot(index="session_date", columns="symbol", values="weight").fillna(0.0)
    r = weights.pivot(index="session_date", columns="symbol", values="fwd_return_1d").fillna(0.0)
    # Align columns (just in case)
    common = w.columns.intersection(r.columns)
    w = w[common].sort_index()
    r = r[common].sort_index()

    # Turnover: |w_t - w_{t-1}| summed across names
    w_prev = w.shift(1).fillna(0.0)
    turnover_per_day = (w - w_prev).abs().sum(axis=1)

    # Slippage cost: one-way on turnover
    slippage_cost = turnover_per_day * (cfg.slippage_bps / 10_000)

    # Borrow cost on shorts (daily rate = annual / 252)
    short_notional = w.clip(upper=0).abs().sum(axis=1)
    borrow_cost = short_notional * (cfg.borrow_bps_annual / 10_000 / 252)

    # Gross P&L: weight · return
    gross_pnl = (w * r).sum(axis=1)
    net_pnl = gross_pnl - slippage_cost - borrow_cost

    equity = (1 + net_pnl).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1

    gross_exp = w.abs().sum(axis=1)
    net_exp = w.sum(axis=1)

    daily = pd.DataFrame({
        "gross_pnl": gross_pnl,
        "slippage_cost": slippage_cost,
        "borrow_cost": borrow_cost,
        "net_pnl": net_pnl,
        "equity": equity,
        "drawdown": drawdown,
        "gross_exposure": gross_exp,
        "net_exposure": net_exp,
        "turnover": turnover_per_day,
    })
    return daily.reset_index()


def _summary_stats(daily: pd.DataFrame) -> dict:
    ret = daily["net_pnl"]
    ret_nz = ret[ret != 0]
    n_days = len(ret)
    n_years = n_days / 252 if n_days > 0 else 1
    total_return = float((1 + ret).prod() - 1)
    annual_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else np.nan
    vol = float(ret.std() * np.sqrt(252))
    sharpe = annual_return / vol if vol > 0 else np.nan
    max_dd = float(daily["drawdown"].min())
    hit = float((ret > 0).mean())
    avg_turnover = float(daily["turnover"].mean())
    gross_avg = float(daily["gross_exposure"].mean())
    net_mean = float(daily["net_exposure"].mean())

    return {
        "n_days": int(n_days),
        "n_years": float(n_years),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "hit_rate_daily": hit,
        "avg_gross_exposure": gross_avg,
        "avg_net_exposure": net_mean,
        "avg_turnover": avg_turnover,
    }


def run_backtest(
    score_col: str = "oof_meta_with_news",
    since: date | None = None,
    cfg: BacktestConfig | None = None,
) -> BacktestResult:
    """End-to-end backtest from OOF predictions."""
    cfg = cfg or BacktestConfig()

    logger.info("Loading backtest inputs (score={})…", score_col)
    raw = _load_backtest_inputs(score_col, since)
    if raw.empty:
        raise RuntimeError("No rows for backtest — did you run `bot8 train quant`?")
    logger.info("  {} rows, {} days, {} symbols",
                len(raw), raw["session_date"].nunique(), raw["symbol"].nunique())

    logger.info("Building daily portfolio weights…")
    weights = build_daily_weights(raw, score_col="score", cfg=cfg.portfolio)

    if cfg.apply_regime:
        logger.info("Applying regime scaling…")
        regime = raw[["session_date", "regime_exposure"]].drop_duplicates("session_date")
        weights = apply_regime_scaling(weights, regime)

    # Join fwd_return_1d back onto the weights for P&L simulation.
    weights = weights.merge(
        raw[["session_date", "symbol", "fwd_return_1d"]],
        on=["session_date", "symbol"],
        how="left",
    )
    weights["fwd_return_1d"] = weights["fwd_return_1d"].fillna(0.0)

    logger.info("Simulating daily P&L…")
    daily = _simulate(weights, cfg)
    summary = _summary_stats(daily)

    logger.info(
        "Summary: {} days, ann.return={:+.2%}, vol={:.2%}, Sharpe={:+.2f}, "
        "DD={:+.2%}, hit={:.1%}, turnover={:.2f}",
        summary["n_days"],
        summary["annual_return"],
        summary["annual_vol"],
        summary["sharpe"],
        summary["max_drawdown"],
        summary["hit_rate_daily"],
        summary["avg_turnover"],
    )

    return BacktestResult(daily=daily, summary=summary, weights=weights)


def save_backtest_result(result: BacktestResult, name: str) -> None:
    """Persist backtest daily + summary to DuckDB for later inspection / dashboard."""
    daily = result.daily.copy()
    daily["backtest_name"] = name
    with session() as con:
        con.register("d", daily)
        con.execute(
            "CREATE TABLE IF NOT EXISTS backtest_daily AS SELECT * FROM d WHERE 1=0"
        )
        con.execute("DELETE FROM backtest_daily WHERE backtest_name = ?", [name])
        con.execute("INSERT INTO backtest_daily SELECT * FROM d")

        summary_df = pd.DataFrame([{**result.summary, "backtest_name": name}])
        con.register("s", summary_df)
        con.execute("CREATE TABLE IF NOT EXISTS backtest_summary AS SELECT * FROM s WHERE 1=0")
        con.execute("DELETE FROM backtest_summary WHERE backtest_name = ?", [name])
        con.execute("INSERT INTO backtest_summary SELECT * FROM s")
    logger.info("Saved backtest '{}' to backtest_daily + backtest_summary", name)
