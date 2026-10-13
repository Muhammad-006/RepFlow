#!/usr/bin/env bash
# RepFlow - build the Android APK (macOS / Linux)
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" tools/write_build_config.py
cd client
if [ ! -f .venv/.installed ]; then
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt "flet-cli>=0.86.0,<0.87.0"
  touch .venv/.installed
fi
echo "[RepFlow] Building APK (first build downloads Flutter + Android SDK: 10-20 min)..."
.venv/bin/flet build apk --yes
echo "Done! APK is in client/build/apk/"
