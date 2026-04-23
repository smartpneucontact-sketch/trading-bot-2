"""Read-only DuckDB connection for the API.

Read-only by design — the UI observes the bot; it never mutates the DB.
Write paths (config updates, pause/resume) go through explicit FastAPI
endpoints that call bot8 business logic, not raw SQL.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import duckdb

from bot8.config import get_settings


@contextmanager
def api_session() -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a read-only DuckDB connection for API request-scoped queries."""
    con = duckdb.connect(database=str(get_settings().db_path), read_only=True)
    try:
        yield con
    finally:
        con.close()
