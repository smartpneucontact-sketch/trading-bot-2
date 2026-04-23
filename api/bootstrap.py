"""One-shot DB bootstrap for the API container.

On Railway we can't `scp` a file onto the persistent volume easily — but the
container has full outbound internet. So: if `DASHBOARD_DB_URL` is set and
the DuckDB file isn't on disk yet, download it on startup. This runs once
per volume lifetime — subsequent container restarts see the file and skip.

Use: host the gzipped dashboard DB on a GitHub Release, HuggingFace, or
S3. Set the full URL as `DASHBOARD_DB_URL` in Railway env vars. Redeploy.
On first boot the API streams the file onto /app/data/db/bot8.duckdb.

Safety:
- No-op if the DB file already exists (fast).
- No-op if `DASHBOARD_DB_URL` is unset (falls back to empty-DB mode).
- Streaming download with gunzip — doesn't need 2x disk space.
- Writes to a temp file first, then atomically renames — so a crashed
  download never leaves a partial `bot8.duckdb` that would confuse DuckDB.
"""

from __future__ import annotations

import gzip
import os
import shutil
import urllib.request
from pathlib import Path

from loguru import logger

from bot8.config import get_settings


def ensure_dashboard_db() -> None:
    """Download + unpack the dashboard DB if it's missing on the volume."""
    s = get_settings()
    db_path = s.db_path
    url = os.getenv("DASHBOARD_DB_URL", "").strip()

    if db_path.exists():
        # Could have been uploaded manually or from a previous boot — leave it.
        logger.info("DB already present at {} ({:.1f} MB) — skipping bootstrap",
                    db_path, db_path.stat().st_size / 1e6)
        return

    if not url:
        logger.warning(
            "DB missing at {} and DASHBOARD_DB_URL unset — API will serve "
            "empty responses. Set DASHBOARD_DB_URL or upload manually.",
            db_path,
        )
        return

    logger.info("Bootstrapping DB from {}", url)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".duckdb.tmp")

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "bot8-bootstrap/1.0"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            total = int(resp.headers.get("Content-Length", 0)) or None
            logger.info(
                "  downloading {} bytes from URL...",
                f"{total:,}" if total else "unknown size",
            )
            if url.endswith(".gz"):
                # Streaming gunzip so we don't need 2× disk for the tmp file.
                with gzip.GzipFile(fileobj=resp) as gz, open(tmp_path, "wb") as out:
                    shutil.copyfileobj(gz, out, length=1024 * 1024)
            else:
                with open(tmp_path, "wb") as out:
                    shutil.copyfileobj(resp, out, length=1024 * 1024)

        # Atomic rename so a crash mid-download leaves no partial bot8.duckdb.
        tmp_path.rename(db_path)
        size_mb = db_path.stat().st_size / 1e6
        logger.info("✓ DB bootstrapped at {} ({:.1f} MB)", db_path, size_mb)

    except Exception as e:
        logger.exception("DB bootstrap failed: {}", e)
        tmp_path.unlink(missing_ok=True)
        # Don't raise — let the API start and serve empty responses gracefully.
