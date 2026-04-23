"""Loguru sink setup for persistent file logging.

Call `setup_file_logging("<service_name>")` at application startup (CLI,
FastAPI app, cron entrypoints). The default stderr sink stays active so
Railway's dashboard keeps working; file sinks are additive.

Design:
- Daily rotation at midnight UTC; old files automatically gzipped.
- 30-day retention by default (configurable via LOG_RETENTION_DAYS).
- Per-service log file name: `{service}_{YYYY-MM-DD}.log` — easy to grep
  across services (e.g. "api" vs "premarket" vs "execution").
- Writes are append-only; safe for multi-process on a shared volume.
"""

from __future__ import annotations

import os

from loguru import logger

from bot8.config import get_settings


def setup_file_logging(service_name: str = "bot8") -> None:
    """Add a rotating file sink. Safe to call multiple times (idempotent via tag)."""
    s = get_settings()
    logs_dir = s.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    retention_days = int(os.getenv("LOG_RETENTION_DAYS", "30"))

    sink_id = f"bot8-file-{service_name}"
    # Remove any prior handler with the same tag so reloads don't duplicate.
    # Loguru doesn't support named handlers natively — we track by adding a
    # tag and filtering.
    for h_id in list(logger._core.handlers):  # type: ignore[attr-defined]
        existing = logger._core.handlers[h_id]  # type: ignore[attr-defined]
        if getattr(existing, "_bot8_tag", None) == sink_id:
            logger.remove(h_id)

    handler_id = logger.add(
        str(logs_dir / f"{service_name}_{{time:YYYY-MM-DD}}.log"),
        rotation="00:00",                 # roll at UTC midnight
        retention=f"{retention_days} days",
        compression="gz",
        level=s.log_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
            "{name}:{function}:{line} - {message}"
        ),
        enqueue=True,                     # thread-safe; fine on shared volume
        backtrace=True,
        diagnose=False,                   # don't leak local vars into logs
    )
    # Tag the handler so we can find and replace it on reload.
    logger._core.handlers[handler_id]._bot8_tag = sink_id  # type: ignore[attr-defined]

    logger.info(
        "File logging enabled: service={} dir={} retention={}d",
        service_name, logs_dir, retention_days,
    )
