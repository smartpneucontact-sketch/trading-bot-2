"""Simple API key auth for the bot8 dashboard.

For a single-user local dashboard this is enough. Upgrade to Clerk or
magic-link auth before exposing on a public URL with real trading writes.

Setup:
- Set DASHBOARD_API_KEY in .env (any string; generate with `openssl rand -hex 32`)
- Frontend sends `X-API-Key: <key>` header on every request
- If DASHBOARD_API_KEY is empty, auth is DISABLED (convenient for local dev)
"""

from __future__ import annotations

from fastapi import Header, HTTPException
from pydantic import SecretStr

from bot8.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency — raises 401 if DASHBOARD_API_KEY is set and doesn't match."""
    s = get_settings()
    expected = _get_dashboard_key()
    if not expected:
        return  # Auth disabled when key is empty.
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _get_dashboard_key() -> str:
    """Read DASHBOARD_API_KEY from settings, or empty if not present."""
    s = get_settings()
    key = getattr(s, "dashboard_api_key", None)
    if isinstance(key, SecretStr):
        return key.get_secret_value()
    return key or ""
