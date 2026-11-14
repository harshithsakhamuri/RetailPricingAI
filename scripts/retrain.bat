@echo off
rem Retrain models, refresh the webpage data (docs\data.js) and re-check every invariant.
cd /d "%~dp0.."
.venv\Scripts\python.exe src\pricing_profit_optimizer.py %* || exit /b 1
.venv\Scripts\python.exe src\export_web_data.py || exit /b 1
.venv\Scripts\python.exe -m pytest -q
