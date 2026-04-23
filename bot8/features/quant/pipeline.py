"""Feature pipeline orchestrator.

Reads from `bars_daily`, `macro_daily`, `universe`. Computes per-symbol +
macro + cross-sectional features. Writes to `features_quant`. Computes
labels into `labels_quant`.

Design:
- Full rebuild is the default. Features are cheap to recompute (< 5 min on
  730K bars, ~600 symbols) and we don't want partial/stale rows mixed in
  after a feature-version bump.
- Tables use CREATE OR REPLACE so the schema auto-evolves when feature
  columns change.
- Sector is pulled from the latest `universe` snapshot and joined in before
  cross-sectional ranking (so rank_within_sector has the data it needs).
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger

from bot8.data.db import session
from bot8.features.quant.cross import (
    DEFAULT_CROSS_RANK_COLS,
    rank_cross_sectional,
    rank_within_sector,
)
from bot8.features.quant.labels import compute_labels
from bot8.features.quant.macro import compute_macro_features
from bot8.features.quant.stock import FEATURE_VERSION, compute_stock_features


def _load_bars(since: date | None, symbols: Iterable[str] | None) -> pd.DataFrame:
    with session(read_only=True) as con:
        sql = """
            SELECT symbol, session_date, open, high, low, close,
                   adj_close, volume
            FROM bars_daily
        """
        clauses: list[str] = []
        params: list = []
        if since:
            clauses.append("session_date >= ?")
            params.append(since)
        if symbols:
            sym_list = list(symbols)
            placeholders = ",".join(["?"] * len(sym_list))
            clauses.append(f"symbol IN ({placeholders})")
            params.extend(sym_list)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY symbol, session_date"
        return con.execute(sql, params).fetchdf()


def _load_universe_sector() -> pd.DataFrame:
    """Latest universe snapshot: one row per symbol → sector.

    Careful: the universe has dual-membership rows (SP500 + NDX100) which can
    disagree on sector naming (SP500 uses GICS, NDX100 uses ICB). We must
    collapse to one row per symbol before merging into the feature frame —
    otherwise the merge fans out the row count.

    Preference order: SP500 (GICS) > NDX100 (ICB).
    """
    with session(read_only=True) as con:
        df = con.execute(
            """
            SELECT symbol, index_membership, sector
            FROM universe
            WHERE as_of_date = (SELECT MAX(as_of_date) FROM universe)
            """
        ).fetchdf()
    if df.empty:
        return df
    # Explicit priority: SP500 first, fall back to NDX100.
    priority = {"SP500": 0, "NDX100": 1}
    df["_p"] = df["index_membership"].map(priority).fillna(99).astype(int)
    df = df.sort_values(["symbol", "_p"]).drop_duplicates(subset=["symbol"], keep="first")
    return df[["symbol", "sector"]].reset_index(drop=True)


def build_features(
    since: date | None = None,
    symbols: Iterable[str] | None = None,
) -> dict[str, int]:
    """End-to-end: compute quant features + labels and write to DB."""
    logger.info("Loading bars…")
    bars = _load_bars(since, symbols)
    if bars.empty:
        logger.warning("No bars found — have you run `bot8 data bars`?")
        return {"features": 0, "labels": 0}
    logger.info("  {:,} rows, {} symbols", len(bars), bars["symbol"].nunique())

    # ---- Per-symbol features -------------------------------------------
    # Iterate groups manually — pandas 3 removed groupby.apply(include_groups=)
    # and the default now drops the grouping column. Manual iteration is
    # simpler and lets us progress-log long runs.
    logger.info("Computing per-symbol features…")
    n_symbols = bars["symbol"].nunique()
    per_symbol_frames: list[pd.DataFrame] = []
    for i, (sym, grp) in enumerate(bars.groupby("symbol", sort=False), 1):
        per_symbol_frames.append(compute_stock_features(grp))
        if i % 50 == 0:
            logger.info("  symbol features: {}/{}", i, n_symbols)
    per_symbol = pd.concat(per_symbol_frames, ignore_index=True)
    # Normalize session_date to python date — DuckDB returns it as such, but
    # some pandas ops coerce to datetime64; we need a consistent type for merges.
    per_symbol["session_date"] = pd.to_datetime(per_symbol["session_date"]).dt.date
    logger.info("  {} per-symbol feature columns", per_symbol.shape[1] - 8)  # minus raw OHLCV cols

    # ---- Macro features -------------------------------------------------
    logger.info("Computing macro features…")
    macro = compute_macro_features(
        since=since.isoformat() if since else None
    )
    if not macro.empty:
        macro["session_date"] = pd.to_datetime(macro["session_date"]).dt.date
        logger.info("  {} macro feature columns", macro.shape[1] - 1)
        per_symbol = per_symbol.merge(macro, on="session_date", how="left")
    else:
        logger.warning("  macro_daily is empty; skipping macro features")

    # ---- Sector join (for within-sector ranks) --------------------------
    sectors = _load_universe_sector()
    if not sectors.empty:
        per_symbol = per_symbol.merge(sectors, on="symbol", how="left")

    # ---- Cross-sectional ranks -----------------------------------------
    logger.info("Computing cross-sectional ranks…")
    cs = rank_cross_sectional(per_symbol, DEFAULT_CROSS_RANK_COLS)
    sector_cs = rank_within_sector(per_symbol, DEFAULT_CROSS_RANK_COLS)
    per_symbol = per_symbol.merge(cs, on=["session_date", "symbol"], how="left")
    per_symbol = per_symbol.merge(sector_cs, on=["session_date", "symbol"], how="left")

    # Drop raw OHLCV columns — they're in bars_daily, no need to duplicate.
    drop_cols = [c for c in ["open", "high", "low", "close", "adj_close", "volume"]
                 if c in per_symbol.columns]
    per_symbol = per_symbol.drop(columns=drop_cols)
    per_symbol["feature_version"] = FEATURE_VERSION

    # ---- Write features -------------------------------------------------
    n_features = _write_features(per_symbol)
    logger.info("Wrote {:,} feature rows ({} cols)", n_features, per_symbol.shape[1])

    # ---- Compute + write labels ----------------------------------------
    logger.info("Computing labels…")
    labels = compute_labels(since=since)
    n_labels = _write_labels(labels)
    logger.info("Wrote {:,} label rows", n_labels)

    return {"features": n_features, "labels": n_labels}


def _write_features(df: pd.DataFrame) -> int:
    """CREATE OR REPLACE features_quant from the DataFrame."""
    if df.empty:
        return 0
    # DuckDB can't handle Python `date` objects in some CREATE TABLE AS paths;
    # ensure session_date is a proper date type.
    df = df.copy()
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    with session() as con:
        con.register("fq", df)
        con.execute("CREATE OR REPLACE TABLE features_quant AS SELECT * FROM fq")
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_fq_symbol_date ON features_quant(symbol, session_date)"
        )
        return con.execute("SELECT COUNT(*) FROM features_quant").fetchone()[0]


def _write_labels(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    with session() as con:
        con.register("lq", df)
        con.execute("CREATE OR REPLACE TABLE labels_quant AS SELECT * FROM lq")
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_lq_symbol_date ON labels_quant(symbol, session_date)"
        )
        return con.execute("SELECT COUNT(*) FROM labels_quant").fetchone()[0]
