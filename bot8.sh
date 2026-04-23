#!/usr/bin/env bash
# Wrapper that invokes the bot8 CLI via `python -m`, bypassing the generated
# `bot8` entry-point script in .venv/bin/. That script depends on a `.pth`
# file that macOS + iCloud Drive aggressively flag as hidden, causing
# Python 3.13's site.py to ignore it.
#
# `python -m bot8.cli` works regardless because we cd into the repo root and
# the current working directory is always on sys.path.
#
# Usage:
#   ./bot8.sh data init
#   ./bot8.sh features news --backfill
#   ./bot8.sh info
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
exec python -m bot8.cli "$@"
