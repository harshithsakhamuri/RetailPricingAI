@echo off
rem Streamlit app. Trains first if no results exist.
cd /d "%~dp0.."
if not exist outputs\app_products.parquet (
    echo First run: training models, about 2 minutes...
    .venv\Scripts\python.exe src\pricing_profit_optimizer.py || exit /b 1
    .venv\Scripts\python.exe src\export_web_data.py || exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
