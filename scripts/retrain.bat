@echo off
rem Retrain models and refresh the webpage data (docs\data.js).
cd /d "%~dp0.."
.venv\Scripts\python.exe src\pricing_profit_optimizer.py
.venv\Scripts\python.exe src\export_web_data.py
