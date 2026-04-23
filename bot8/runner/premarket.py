"""Pre-market runner — daily orchestrator that turns scored data into orders.

Timing (US Eastern, ET):
    06:30  Railway cron fires `bot8 live premarket`
    06:30  Check market-open-today + holidays; exit if closed.
    06:32  Fetch live news (Alpaca News + Yahoo, incremental).
    06:40  Score new headlines (FinBERT + regex).
    06:42  Aggregate per-(symbol, date) news features.
    06:45  Refresh daily bars (for features that need today's open gap).
    06:50  Run trained base models on today's features.
    06:55  Run meta-learner → final {symbol: score}.
    07:00  Build target portfolio (longs/shorts, conviction-weighted, capped).
    09:35  Submit rebalance orders to Alpaca.
    09:40  Poll for fills, emit trade journal entry.

This file is the outer loop — each step below is a thin wrapper around
existing bot8 primitives. When a step isn't yet ready for live data (e.g.
we haven't wired today's features yet), it falls back to reading the most
recent OOF predictions and uses those. This keeps the live loop runnable
end-to-end from day one; we tighten each step as the underlying data
pipeline matures.

Safety:
- Paper trading by default. Live only with explicit --live flag.
- Every step emits a log line. Every failure is caught and reported in
  the final RunReport rather than crashing the process.
- Dry-run mode (--dry-run) computes weights + order plan but does NOT
  submit. Use this for smoke tests against the Alpaca paper account.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
from loguru import logger

from bot8.data.db import session


@dataclass(slots=True)
class PremarketReport:
    """Structured result of one pre-market run — written to trade journal."""
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    market_open: bool = False
    steps: list[dict[str, Any]] = field(default_factory=list)
    target_weights: dict[str, float] = field(default_factory=dict)
    rebalance_nav: Decimal | None = None
    n_longs: int = 0
    n_shorts: int = 0
    n_orders_submitted: int = 0
    n_orders_filled: int = 0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False
    paper: bool = True

    def step(self, name: str, **details: Any) -> None:
        """Log a step result to the report + loguru."""
        self.steps.append({"step": name, "ts": datetime.utcnow().isoformat(), **details})
        logger.info("[premarket/{}] {} {}", self.run_id, name, details)

    def fail(self, step: str, error: str) -> None:
        self.errors.append(f"{step}: {error}")
        logger.error("[premarket/{}] {} FAILED: {}", self.run_id, step, error)


# ---------------------------------------------------------------------------
# Market calendar — cheap check to skip weekends and major US holidays
# ---------------------------------------------------------------------------

US_MARKET_HOLIDAYS_2026 = {
    date(2026, 1, 1),   # New Year
    date(2026, 1, 19),  # MLK
    date(2026, 2, 16),  # Presidents
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # July 4th (observed)
    date(2026, 9, 7),   # Labor
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
}


def is_trading_day(d: date) -> bool:
    """True if d is a NYSE/NASDAQ trading day (Mon-Fri, not a major holiday)."""
    if d.weekday() >= 5:
        return False
    if d in US_MARKET_HOLIDAYS_2026:
        return False
    return True


# ---------------------------------------------------------------------------
# Target-weight construction from predictions
# ---------------------------------------------------------------------------


def load_latest_predictions(score_col: str = "oof_meta_with_news") -> dict[str, float]:
    """Most-recent {symbol: score} from quant_oof_preds.

    This is a TEMPORARY bridge — ideally we'd recompute predictions on
    fresh features each morning. For now, the live loop uses the backtest
    predictions as a stand-in so the execution path can be tested end to end.
    Replace with a real 'score today's features' call when that pipeline is wired.
    """
    with session(read_only=True) as con:
        rows = con.execute(
            f"""
            SELECT symbol, {score_col} AS score
            FROM quant_oof_preds
            WHERE session_date = (SELECT MAX(session_date) FROM quant_oof_preds)
              AND {score_col} IS NOT NULL
            ORDER BY score DESC
            """
        ).fetchall()
    return {sym: float(s) for sym, s in rows}


def build_target_weights_from_scores(
    scores: dict[str, float],
    shortable_filter: set[str] | None = None,
    long_decile: float = 0.10,
    short_decile: float = 0.10,
    per_name_cap: float = 0.02,
) -> dict[str, float]:
    """Simple decile-based long/short weighting — mirrors the backtest logic.

    - Long top decile, short bottom decile (filtered to shortable only)
    - Conviction-weighted within each side, then per-name cap
    - Returns {symbol: weight} with sum ≈ 0 (dollar neutral)
    """
    if not scores:
        return {}

    df = pd.DataFrame({"symbol": list(scores.keys()), "score": list(scores.values())})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    n = len(df)
    n_long = max(1, int(n * long_decile))
    n_short = max(1, int(n * short_decile))

    df["rank_pct"] = df.index / max(1, n - 1)
    df["side"] = "flat"
    df.loc[: n_long - 1, "side"] = "long"

    short_candidates = df.iloc[-n_short * 3 :].sort_values("score", ascending=True)
    if shortable_filter is not None:
        short_candidates = short_candidates[short_candidates["symbol"].isin(shortable_filter)]
    short_syms = set(short_candidates.head(n_short)["symbol"])
    df.loc[df["symbol"].isin(short_syms), "side"] = "short"

    weights: dict[str, float] = {}
    # Long side
    longs = df[df["side"] == "long"]
    if len(longs) > 0:
        raw = (longs["score"] - longs["score"].min() + 1e-6).values
        raw = raw / raw.sum()  # normalize to sum 1
        for sym, w in zip(longs["symbol"], raw):
            weights[sym] = min(float(w), per_name_cap)
    # Short side
    shorts = df[df["side"] == "short"]
    if len(shorts) > 0:
        raw = (shorts["score"].abs() + 1e-6).values
        raw = raw / raw.sum()
        for sym, w in zip(shorts["symbol"], raw):
            weights[sym] = -min(float(w), per_name_cap)

    return weights


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_premarket(
    score_col: str = "oof_meta_with_news",
    dry_run: bool = False,
    paper: bool = True,
    skip_news_fetch: bool = False,
) -> PremarketReport:
    """Orchestrates the full pre-market flow. Returns a structured report."""
    today = date.today()
    run_id = today.strftime("%Y%m%d") + datetime.utcnow().strftime("%H%M%S")
    report = PremarketReport(
        run_id=run_id,
        started_at=datetime.utcnow(),
        dry_run=dry_run,
        paper=paper,
    )

    # 0. Trading day check
    if not is_trading_day(today):
        report.step("market_closed", date=str(today))
        report.finished_at = datetime.utcnow()
        return report
    report.market_open = True
    report.step("market_open_check", date=str(today))

    # 1. Fetch live news (can be skipped when we already have today's data)
    if not skip_news_fetch:
        try:
            from bot8.data.news.live import fetch_alpaca_news_incremental
            universe_syms = _get_tradable_universe()
            n = fetch_alpaca_news_incremental(universe_syms, lookback_days=3)
            report.step("fetch_news", source="alpaca", new_rows=n)
        except Exception as e:
            report.fail("fetch_news", str(e))

    # 2. Score new headlines (FinBERT + regex)
    try:
        from bot8.features.news.scorer import score_batch
        summary = score_batch(batch_size=256)
        report.step("score_news", rows_scored=summary.total_scored,
                    seconds=summary.seconds)
    except Exception as e:
        report.fail("score_news", str(e))

    # 3. Aggregate per-day news features
    try:
        from bot8.features.news.aggregator import build_news_features
        n = build_news_features()
        report.step("aggregate_news", rows=n)
    except Exception as e:
        report.fail("aggregate_news", str(e))

    # 4. Load today's predictions (TEMP: uses latest OOF; see TODO in
    #    load_latest_predictions — replace with fresh feature-compute + model
    #    inference once the pipeline is wired for same-day data).
    try:
        scores = load_latest_predictions(score_col=score_col)
        report.step("load_predictions", score_col=score_col, n_symbols=len(scores))
    except Exception as e:
        report.fail("load_predictions", str(e))
        scores = {}

    # 5. Build target portfolio weights
    if scores:
        shortable = _get_shortable_set()
        weights = build_target_weights_from_scores(
            scores=scores,
            shortable_filter=shortable,
        )
        report.target_weights = weights
        report.n_longs = sum(1 for w in weights.values() if w > 0)
        report.n_shorts = sum(1 for w in weights.values() if w < 0)
        report.step("build_weights",
                    n_longs=report.n_longs,
                    n_shorts=report.n_shorts,
                    gross=sum(abs(w) for w in weights.values()))

    # 6. Submit orders to Alpaca (skipped in dry-run)
    if dry_run or not report.target_weights:
        report.step("execute", skipped_reason=("dry_run" if dry_run else "no_weights"))
    else:
        try:
            from bot8.execution.alpaca import AlpacaExecutor
            executor = AlpacaExecutor(paper=paper)
            report.rebalance_nav = executor.get_nav()
            rebalance_report = executor.rebalance(
                target_weights=report.target_weights,
                run_id=run_id,
                cancel_open_first=True,
                wait_for_fills=True,
            )
            report.n_orders_submitted = len(rebalance_report.submitted_order_ids)
            report.n_orders_filled = rebalance_report.n_filled
            report.step("execute",
                        nav=str(report.rebalance_nav),
                        submitted=report.n_orders_submitted,
                        filled=report.n_orders_filled,
                        duration_s=f"{rebalance_report.duration_s:.1f}")
        except Exception as e:
            report.fail("execute", str(e))

    report.finished_at = datetime.utcnow()
    return report


# ---------------------------------------------------------------------------
# DB helpers — small read-only lookups for universe and shortable flags
# ---------------------------------------------------------------------------


def _get_tradable_universe() -> list[str]:
    """Current tradable universe from the most recent snapshot."""
    with session(read_only=True) as con:
        rows = con.execute(
            """
            SELECT DISTINCT symbol FROM universe
            WHERE as_of_date = (SELECT MAX(as_of_date) FROM universe)
              AND is_tradable = TRUE
            ORDER BY symbol
            """
        ).fetchall()
    return [r[0] for r in rows]


def _get_shortable_set() -> set[str]:
    """Set of symbols marked shortable on the latest universe snapshot."""
    with session(read_only=True) as con:
        rows = con.execute(
            """
            SELECT DISTINCT symbol FROM universe
            WHERE as_of_date = (SELECT MAX(as_of_date) FROM universe)
              AND is_shortable = TRUE
            """
        ).fetchall()
    return {r[0] for r in rows}
