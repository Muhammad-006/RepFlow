#!/usr/bin/env bash
# RepFlow AI Coach Server - macOS / Linux launcher
set -e
cd "$(dirname "$0")/server"
PY="${PYTHON:-python3}"
if [ ! -f .venv/.installed ]; then
  echo "[RepFlow] First run - setting up the AI server (one time, ~1-2 minutes)..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.installed
fi
exec .venv/bin/python server.py "$@"
