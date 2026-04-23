"""Shared router helpers."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import duckdb
from loguru import logger


def safe_query(default_factory: Callable[[], Any]):
    """Decorator: run an endpoint, return default_factory() on
    "table doesn't exist yet" style errors. Keeps the dashboard usable before
    the DuckDB file has been uploaded to the Railway volume."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except duckdb.CatalogException as e:
                logger.debug("safe_query fallback for {}: {}", fn.__name__, e)
                return default_factory()
        return wrapped
    return decorator
