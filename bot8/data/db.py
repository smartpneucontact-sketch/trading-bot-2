"""DuckDB connection + schema bootstrap.

DuckDB chosen because:
- Embedded (no server), great for local iteration + Railway deploy.
- Columnar, fast for the analytical queries we'll run (per-symbol time-series aggregations).
- Native Parquet / Pandas / Arrow interop.
- Single-file DB ships cleanly in the Railway volume.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb
from loguru import logger

from bot8.config import get_settings

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection at the configured path.

    Callers are responsible for closing. For short-lived work prefer `session()`.
    """
    s = get_settings()
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=str(s.db_path), read_only=read_only)
    return con


@contextmanager
def session(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Context-managed connection. Commits on exit (write mode)."""
    con = connect(read_only=read_only)
    try:
        yield con
        if not read_only:
            con.commit()
    finally:
        con.close()


def init_schema() -> None:
    """Execute schema.sql. Idempotent."""
    sql = _SCHEMA_PATH.read_text()
    with session() as con:
        con.execute(sql)
    logger.info("DuckDB schema initialised at {}", get_settings().db_path)
