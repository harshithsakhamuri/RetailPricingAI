#!/usr/bin/env sh
# Streamlit app. Trains first if no results exist.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
if [ ! -f outputs/app_products.parquet ]; then
  echo "First run: training models, about 2 minutes..."
  "$PY" src/pricing_profit_optimizer.py
  "$PY" src/export_web_data.py
fi
"$PY" -m streamlit run app/streamlit_app.py
