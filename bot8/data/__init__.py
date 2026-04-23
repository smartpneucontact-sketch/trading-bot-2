"""Data layer — DuckDB connection, raw news + market ingestion."""

from bot8.data.db import connect, init_schema

__all__ = ["connect", "init_schema"]
