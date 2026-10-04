# Run FastAPI Backend Server
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host " Starting AUTOSAR HLD Analysis Backend (FastAPI)     " -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Cyan

$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
