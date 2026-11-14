#!/usr/bin/env sh
# Retrain models, refresh the webpage data and re-check every invariant.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
"$PY" src/pricing_profit_optimizer.py "$@"
"$PY" src/export_web_data.py
"$PY" -m pytest -q
