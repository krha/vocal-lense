#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
APP_FILE="$ROOT_DIR/web_app.py"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Error: virtual environment not found at $VENV_DIR" >&2
  echo "Create it with: python3 -m venv .venv" >&2
  exit 1
fi

if [[ ! -f "$APP_FILE" ]]; then
  echo "Error: web app not found at $APP_FILE" >&2
  exit 1
fi

cd "$ROOT_DIR"
source "$VENV_DIR/bin/activate"
exec python "$APP_FILE"
