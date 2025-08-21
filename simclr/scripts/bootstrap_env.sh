#!/usr/bin/env bash
# Optional: create and sync Python environment with uv.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Please install uv first: https://github.com/astral-sh/uv"
  exit 1
fi

uv venv -p 3.11 .venv
uv sync --python 3.11

echo "Environment ready. Activate with: source .venv/bin/activate"
