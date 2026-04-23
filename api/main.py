"""FastAPI application entry point.

Run locally:  uvicorn api.main:app --reload --port 8000
Production:   uvicorn api.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.auth import require_api_key
from api.routers import backtest, news, portfolio, trades

app = FastAPI(
    title="bot8 API",
    version="0.1.0",
    description="Read-only REST API over the bot8 DuckDB for the web dashboard.",
)

# CORS origins: localhost for dev + env-supplied list for production.
# FRONTEND_ORIGINS=https://bot8.vercel.app,https://bot8-preview.vercel.app
_env_origins = os.getenv("FRONTEND_ORIGINS", "").split(",")
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    *[o.strip() for o in _env_origins if o.strip()],
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# All /api/* endpoints protected by API-key auth. /health and / stay public.
api_deps = [Depends(require_api_key)]
app.include_router(portfolio.router, dependencies=api_deps)
app.include_router(trades.router, dependencies=api_deps)
app.include_router(backtest.router, dependencies=api_deps)
app.include_router(news.router, dependencies=api_deps)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "bot8-api", "version": app.version}


@app.get("/")
def root() -> dict:
    return {
        "name": "bot8 API",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/api/portfolio",
            "/api/equity-curve",
            "/api/metrics",
            "/api/trades",
            "/api/backtest/compare",
            "/api/news/recent",
        ],
    }
