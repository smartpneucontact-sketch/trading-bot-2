"""Read-only DuckDB connection for the API.

Read-only by design — the UI observes the bot; it never mutates the DB.
Write paths (config updates, pause/resume) go through explicit FastAPI
endpoints that call bot8 business logic, not raw SQL.

Before the DuckDB file has been uploaded to the Railway volume (fresh deploy),
we fall back to an in-memory connection so endpoints can still return empty
results cleanly instead of 500ing. Frontend shows empty tables rather than
broken cards.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import duckdb
from loguru import logger

from bot8.config import get_settings


@contextmanager
def api_session() -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a read-only DuckDB connection, or an empty in-memory one if the
    file hasn't been uploaded yet."""
    db_path = get_settings().db_path
    if db_path.exists():
        con = duckdb.connect(database=str(db_path), read_only=True)
    else:
        logger.warning("DuckDB file not found at {} — serving empty responses", db_path)
        con = duckdb.connect(database=":memory:", read_only=False)
    try:
        yield con
    except duckdb.CatalogException:
        # Table doesn't exist yet (e.g. backtest never run). Let the endpoint
        # decide how to present this — usually an empty list.
        raise
    finally:
        con.close()
