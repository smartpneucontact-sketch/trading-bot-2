# Dockerfile for the bot8 FastAPI service.
# Build context is the repo root.
#
# Railway deployment:
#   1. Add this repo as a Railway service
#   2. Set Dockerfile path: Dockerfile.api
#   3. Env vars:
#        ALPACA_API_KEY, ALPACA_SECRET_KEY
#        DASHBOARD_API_KEY (openssl rand -hex 32)
#        FRONTEND_ORIGINS=https://<vercel-url>
#        DATA_DIR=/app/data
#   4. Add persistent volume at /app/data (for DuckDB + models)

# Python 3.12 — wider wheel coverage than 3.13 for duckdb/pyarrow/lxml on
# Railway's x86_64 build env. Swap to 3.13 once upstream wheels catch up.
FROM python:3.12-slim

# System deps for duckdb, numpy, pandas wheels.
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching.
#
# IMPORTANT — keep this install minimal. The API is a read-only DuckDB
# consumer; it does NOT need alpaca-py, yfinance, or lxml at this stage.
# The full trading extra triples the install size and has caused OOM kills
# on Railway's standard builder. When we wire in the live execution adapter,
# move it to its own service/image rather than bloating this one.
COPY pyproject.toml README.md ./
COPY bot8/ ./bot8/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[web]"

# Copy the API code last so it rebuilds cheaply.
COPY api/ ./api/

# Runtime.
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data
# Optional: mount a separate Railway volume at /app/logs and set LOGS_DIR
# to split logs from the DuckDB. Falls back to /app/data/logs when unset.
# ENV LOGS_DIR=/app/logs
EXPOSE 8000

# Railway sets $PORT; fall back to 8000 for local.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
