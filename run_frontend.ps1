# Run Streamlit Frontend UI
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host " Starting AUTOSAR HLD Dashboard (Streamlit)          " -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Cyan

$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m streamlit run streamlit_app/app.py --server.port 8501
