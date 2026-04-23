"""Typer CLI entry point. `bot8 <command> <subcommand>`.

Design: thin wrappers that call into module-level functions. No business logic here.
Makes the CLI testable (business functions are importable) and keeps argument
parsing separate from domain code.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer
from loguru import logger

from bot8.config import get_settings
from bot8.config.logging import setup_file_logging

# Every CLI invocation writes to the "cli" log file. Useful for auditing
# cron-driven runs on Railway (bot8 data fnspid, bot8 train quant, etc).
setup_file_logging(service_name="cli")

app = typer.Typer(
    name="bot8",
    help="Hybrid quant + news trading bot. Paper-first, Alpaca, daily rebalance.",
    no_args_is_help=True,
)

data_app = typer.Typer(help="Data ingestion commands.")
app.add_typer(data_app, name="data")

features_app = typer.Typer(help="Feature engineering commands.")
app.add_typer(features_app, name="features")


@app.command()
def info() -> None:
    """Print resolved config paths + quick DB status."""
    s = get_settings()
    typer.echo(f"data_dir:     {s.data_dir}")
    typer.echo(f"db_path:      {s.db_path}")
    typer.echo(f"models_dir:   {s.models_dir}")
    typer.echo(f"fnspid_dir:   {s.fnspid_dir}")
    typer.echo(f"alpaca URL:   {s.alpaca_base_url}")
    typer.echo(f"finbert:      {s.finbert_model}")

    if s.db_path.exists():
        from bot8.data.db import session
        with session(read_only=True) as con:
            tables = con.execute("SHOW TABLES").fetchall()
            typer.echo(f"\ntables: {[t[0] for t in tables]}")
            for (tbl,) in tables:
                n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                typer.echo(f"  {tbl:20s} {n:>12,} rows")
    else:
        typer.echo("\n(DuckDB file does not exist yet — run `bot8 data init`)")


@data_app.command("init")
def data_init() -> None:
    """Create the DuckDB file and apply the schema (idempotent)."""
    from bot8.data.db import init_schema
    init_schema()
    typer.echo(f"Schema initialised at {get_settings().db_path}")


@data_app.command("fnspid")
def data_fnspid(
    since: str = typer.Option(
        "2020-01-01",
        help="Only ingest headlines on/after this ISO date (YYYY-MM-DD).",
    ),
    download: bool = typer.Option(True, help="Download from HuggingFace first."),
    file: list[str] = typer.Option(
        None,
        help="Optional allowlist of FNSPID file patterns (e.g. 'Stock_news/nasdaq_*.csv'). "
             "Repeat flag for multiple.",
    ),
) -> None:
    """Download + ingest FNSPID historical news into DuckDB.

    Default skips pre-2020 data to keep the pilot small; use --since 1999-01-01
    for the full 8-year history.
    """
    from bot8.data.news.fnspid_loader import ingest_all

    dt = datetime.fromisoformat(since)
    typer.echo(f"Ingesting FNSPID since {dt.date()} (download={download})…")
    results = ingest_all(since=dt, download=download, files_allowlist=file)

    total = sum(results.values())
    typer.echo(f"\nDone. {total:,} new rows across {len(results)} files:")
    for name, n in sorted(results.items()):
        typer.echo(f"  {name:40s} {n:>10,}")


@data_app.command("universe")
def data_universe() -> None:
    """Fetch S&P 500 + Nasdaq 100 from Wikipedia, enrich with Alpaca metadata,
    write as-of snapshot into the `universe` table."""
    from bot8.data.market.universe import refresh_universe

    n = refresh_universe()
    typer.echo(f"Universe refreshed: {n} rows")


@data_app.command("bars")
def data_bars(
    since: str = typer.Option(
        "2019-01-01",
        help="Start date for backfill. Default 2019-01-01 gives 1yr warmup before 2020 news.",
    ),
    end: str | None = typer.Option(None, help="End date (ISO). Defaults to today."),
    symbols: str | None = typer.Option(
        None,
        help="Comma-separated ticker list. If omitted, uses current universe (S&P 500 + NDX100).",
    ),
    incremental: bool = typer.Option(
        True,
        help="Only fetch dates after the last bar already in DB per symbol.",
    ),
) -> None:
    """Backfill Alpaca daily bars into `bars_daily`."""
    from bot8.data.market.bars import fetch_bars
    from bot8.data.market.universe import current_universe

    start_dt = datetime.fromisoformat(since).date()
    end_dt = datetime.fromisoformat(end).date() if end else None

    if symbols:
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        sym_list = current_universe()
        if not sym_list:
            typer.echo("Universe is empty — run `bot8 data universe` first.")
            raise typer.Exit(1)

    results = fetch_bars(sym_list, start=start_dt, end=end_dt, incremental=incremental)
    total = sum(n for n in results.values() if n > 0)
    failed = [s for s, n in results.items() if n < 0]
    typer.echo(f"\n  bars inserted: {total:,}")
    typer.echo(f"  symbols:       {len(sym_list)} ({len(failed)} failed)")
    if failed:
        typer.echo(f"  failed:        {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")


@data_app.command("macro")
def data_macro(
    since: str = typer.Option("2019-01-01", help="Start date ISO."),
    end: str | None = typer.Option(None, help="End date ISO, defaults to today."),
    series: str | None = typer.Option(
        None,
        help="Comma-separated series codes (VIX,SPY,HYG,...). Default: all known series.",
    ),
) -> None:
    """Backfill macro series (VIX/SPY/HYG/yields/etc.) into `macro_daily`."""
    from bot8.data.market.macro import fetch_macro

    start_dt = datetime.fromisoformat(since).date()
    end_dt = datetime.fromisoformat(end).date() if end else None
    series_list = [s.strip().upper() for s in series.split(",")] if series else None

    results = fetch_macro(start=start_dt, end=end_dt, series=series_list)
    total = sum(n for n in results.values() if n > 0)
    typer.echo(f"\n  macro rows inserted: {total:,}")
    for code, n in sorted(results.items()):
        typer.echo(f"    {code:<6s} {n:>8,}" if n >= 0 else f"    {code:<6s} FAILED")


@features_app.command("quant")
def features_quant(
    since: str = typer.Option(
        "2019-01-01",
        help="Start date for feature computation. Default matches bars backfill.",
    ),
    symbols: str | None = typer.Option(
        None,
        help="Comma-separated ticker list. Defaults to all symbols in bars_daily.",
    ),
) -> None:
    """Compute per-symbol + macro + cross-sectional quant features, plus 1d labels.

    Rebuilds `features_quant` and `labels_quant` tables from scratch (CREATE OR
    REPLACE) — safe to re-run, always produces a consistent snapshot.
    """
    from bot8.features.quant.pipeline import build_features

    since_dt = datetime.fromisoformat(since).date()
    sym_list = (
        [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if symbols
        else None
    )

    result = build_features(since=since_dt, symbols=sym_list)
    typer.echo("")
    typer.echo(f"  features_quant: {result['features']:,} rows")
    typer.echo(f"  labels_quant:   {result['labels']:,} rows")


@features_app.command("news-claude")
def features_news_claude(
    action: str = typer.Argument(
        ...,
        help="One of: estimate | submit | poll | ingest | all (sequential pipeline)",
    ),
    since: str = typer.Option("2022-01-01", help="Window start (ISO)."),
    until: str = typer.Option("2023-12-31", help="Window end (ISO)."),
    top_n: int = typer.Option(200, help="Top-N tickers by news volume in window."),
    max_cost: float = typer.Option(
        50.0,
        help="Hard cost cap in USD. --submit refuses if estimate exceeds this.",
    ),
    batch_id: str | None = typer.Option(
        None,
        help="For --poll / --ingest: the batch ID (defaults to most-recent job in DB).",
    ),
    poll_interval: int = typer.Option(60, help="Seconds between --poll checks."),
    job_name: str = typer.Option(
        "pilot",
        help="Human-readable tag stored with the batch job record.",
    ),
) -> None:
    """Claude-based news scoring pilot via Message Batches API.

    Typical flow:
        ./bot8.sh features news-claude estimate --since 2022-01-01 --until 2023-12-31
        ./bot8.sh features news-claude submit   --since 2022-01-01 --until 2023-12-31 --max-cost 50
        ./bot8.sh features news-claude poll     --batch-id msgbatch_...
        ./bot8.sh features news-claude ingest   --batch-id msgbatch_...

    Or one-shot:
        ./bot8.sh features news-claude all --since 2022-01-01 --until 2023-12-31
    """
    from bot8.features.news.claude_scorer import (
        build_requests,
        estimate_cost,
        ingest_results,
        poll_batch,
        submit_batch,
    )
    from bot8.data.db import session as db_session

    since_d = datetime.fromisoformat(since).date()
    until_d = datetime.fromisoformat(until).date()

    def _latest_batch_id() -> str:
        with db_session(read_only=True) as con:
            row = con.execute(
                "SELECT batch_id FROM claude_batch_jobs ORDER BY submitted_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            raise typer.BadParameter("No batch jobs found. Run `submit` first, or pass --batch-id.")
        return row[0]

    def _all_pending_batch_ids() -> list[str]:
        """All batch IDs that haven't been ingested yet."""
        with db_session(read_only=True) as con:
            rows = con.execute(
                """
                SELECT batch_id FROM claude_batch_jobs
                WHERE ingested_at IS NULL
                ORDER BY submitted_at ASC
                """
            ).fetchall()
        return [r[0] for r in rows]

    if action in ("estimate", "submit", "all"):
        requests = build_requests(since=since_d, until=until_d, top_n_tickers=top_n)
        est = estimate_cost(requests)
        typer.echo("")
        typer.echo("=== Cost estimate ===")
        typer.echo(est.pretty())

        if est.total_cost > max_cost:
            typer.echo(
                f"\n❌ Estimated cost ${est.total_cost:.2f} exceeds --max-cost "
                f"${max_cost:.2f}. Narrow the window or raise the cap."
            )
            raise typer.Exit(1)

        if action == "estimate":
            typer.echo(f"\n✓ Under the ${max_cost:.2f} cap. Run `submit` to proceed.")
            return

        typer.echo(f"\n✓ Within ${max_cost:.2f} cap. Submitting…")
        batch_ids = submit_batch(requests, job_name=job_name)
        typer.echo(f"\nSubmitted {len(batch_ids)} batch(es):")
        for bid in batch_ids:
            typer.echo(f"  {bid}")

        if action == "submit":
            typer.echo("\nUse `poll` (no --batch-id needed) to wait for all pending batches.")
            return

        # action == "all" → continue through poll + ingest for each
        for bid in batch_ids:
            typer.echo(f"\nPolling {bid}…")
            poll_batch(bid, poll_interval_s=poll_interval)
            typer.echo(f"Ingesting {bid}…")
            n = ingest_results(bid)
            typer.echo(f"  ✓ {n:,} rows from {bid}")
        return

    if action == "poll":
        bids = [batch_id] if batch_id else _all_pending_batch_ids()
        if not bids:
            typer.echo("No pending batches to poll.")
            return
        for bid in bids:
            result = poll_batch(bid, poll_interval_s=poll_interval)
            typer.echo(f"Batch {bid} → {result}")
        return

    if action == "ingest":
        bids = [batch_id] if batch_id else _all_pending_batch_ids()
        if not bids:
            typer.echo("No pending batches to ingest.")
            return
        total = 0
        for bid in bids:
            n = ingest_results(bid)
            typer.echo(f"  ✓ {n:,} rows from {bid}")
            total += n
        typer.echo(f"\n✓ Total: {total:,} rows in news_features_claude_daily")
        return

    typer.echo(f"Unknown action: {action}")
    raise typer.Exit(1)


@features_app.command("news-daily")
def features_news_daily(
    since: str | None = typer.Option(
        None,
        help="Only aggregate headlines on/after this date (YYYY-MM-DD).",
    ),
) -> None:
    """Aggregate `news_scored` into per-(symbol,date) news features.

    Output table: `news_features_daily`. Joins cleanly onto `features_quant`
    for the meta-learner.
    """
    from bot8.features.news.aggregator import build_news_features

    since_dt = datetime.fromisoformat(since).date() if since else None
    n = build_news_features(since=since_dt)
    typer.echo(f"\n  news_features_daily: {n:,} rows")


@features_app.command("news")
def features_news(
    backfill: bool = typer.Option(
        False,
        "--backfill",
        help="Score every unscored headline in news_raw (resumable).",
    ),
    since: str | None = typer.Option(
        None,
        help="Only score headlines published on/after this ISO date (YYYY-MM-DD).",
    ),
    limit: int | None = typer.Option(
        None,
        help="Stop after scoring this many headlines (useful for pilots / tests).",
    ),
    batch_size: int = typer.Option(
        256,
        help="DB fetch batch size. FinBERT has its own inference batch (32) inside.",
    ),
) -> None:
    """Score headlines with FinBERT sentiment + regex catalyst classification.

    Writes to `news_scored`. Safe to re-run: only unscored rows at the current
    model version are processed.
    """
    if not backfill and since is None and limit is None:
        typer.echo("Pass --backfill to score everything, or --since / --limit for partial.")
        raise typer.Exit(1)

    from bot8.features.news.scorer import score_batch

    since_dt = datetime.fromisoformat(since) if since else None
    summary = score_batch(batch_size=batch_size, limit=limit, since=since_dt)

    typer.echo("")
    typer.echo(f"  rows scored:  {summary.total_scored:,}")
    typer.echo(f"  batches:      {summary.batches}")
    typer.echo(f"  elapsed:      {summary.seconds:.1f}s")
    typer.echo(f"  throughput:   {summary.rows_per_second:.1f} rows/s")
    typer.echo(f"  version:      {summary.model_version}")


train_app = typer.Typer(help="Model training commands.")
app.add_typer(train_app, name="train")

live_app = typer.Typer(help="Live trading commands — run-loops that submit real orders.")
app.add_typer(live_app, name="live")


@live_app.command("premarket")
def live_premarket(
    score: str = typer.Option("oof_meta_with_news", help="Which OOF meta to trade."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Compute weights but don't submit."),
    paper: bool = typer.Option(True, help="Paper trading (default). Use --no-paper for live."),
    skip_news: bool = typer.Option(False, "--skip-news", help="Skip the live news fetch step."),
) -> None:
    """Daily pre-market run: news fetch -> score -> predict -> rebalance -> report."""
    from bot8.runner.premarket import run_premarket

    report = run_premarket(
        score_col=score,
        dry_run=dry_run,
        paper=paper,
        skip_news_fetch=skip_news,
    )
    typer.echo("")
    typer.echo(f"  run_id:        {report.run_id}")
    typer.echo(f"  market_open:   {report.market_open}")
    typer.echo(f"  longs/shorts:  {report.n_longs}/{report.n_shorts}")
    typer.echo(f"  submitted:     {report.n_orders_submitted}")
    typer.echo(f"  filled:        {report.n_orders_filled}")
    if report.errors:
        typer.echo(f"  ERRORS ({len(report.errors)}):")
        for err in report.errors:
            typer.echo(f"    • {err}")


@train_app.command("quant")
def train_quant_cmd(
    since: str | None = typer.Option(None, help="Only train on data >= this date."),
    n_splits: int = typer.Option(6, help="Number of walk-forward CV folds."),
    embargo_days: int = typer.Option(5, help="Embargo window around each fold boundary."),
    fast: bool = typer.Option(
        False,
        help="Use fast 3-model ensemble instead of full 5-model stack (quicker).",
    ),
) -> None:
    """Train the quant stacked ensemble. Writes OOF predictions + model artifacts."""
    from bot8.models.quant.train import train_quant

    since_dt = datetime.fromisoformat(since).date() if since else None
    report = train_quant(
        since=since_dt,
        n_splits=n_splits,
        embargo_days=embargo_days,
        fast=fast,
    )
    typer.echo("")
    typer.echo(report.pretty())


@app.command()
def backtest(
    score: str = typer.Option(
        "oof_meta_with_news",
        help="Which OOF score to use: oof_meta_with_news | oof_meta_quant_only | oof_lgbm_reg | ...",
    ),
    since: str | None = typer.Option(None, help="Only simulate from this date onwards."),
    compare: bool = typer.Option(
        False,
        help="Run both quant-only and with-news backtests side by side.",
    ),
    slippage_bps: float = typer.Option(5.0, help="One-way slippage in bps applied on turnover."),
    borrow_bps: float = typer.Option(25.0, help="Annual short borrow cost in bps."),
    no_regime: bool = typer.Option(False, help="Disable regime-scaling overlay."),
) -> None:
    """Simulate the long/short strategy on OOF predictions."""
    from bot8.portfolio.longshort import PortfolioConfig
    from bot8.runner.backtest import (
        BacktestConfig,
        run_backtest,
        save_backtest_result,
    )

    since_dt = datetime.fromisoformat(since).date() if since else None

    def _run(score_col: str) -> None:
        cfg = BacktestConfig(
            slippage_bps=slippage_bps,
            borrow_bps_annual=borrow_bps,
            apply_regime=not no_regime,
            portfolio=PortfolioConfig(),
        )
        result = run_backtest(score_col=score_col, since=since_dt, cfg=cfg)
        save_backtest_result(result, name=score_col)
        typer.echo(f"\n=== Backtest: {score_col} ===")
        for k, v in result.summary.items():
            if isinstance(v, float):
                typer.echo(f"  {k:<22s} {v:+.4f}")
            else:
                typer.echo(f"  {k:<22s} {v}")

    if compare:
        _run("oof_meta_quant_only")
        _run("oof_meta_with_news")
    else:
        _run(score)


if __name__ == "__main__":
    app()
