# Start both Backend and Frontend in separate PowerShell processes
Write-Host "Starting AUTOSAR HLD Document Analysis Assistant..." -ForegroundColor Cyan

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"
Start-Sleep -Seconds 3
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; .\.venv\Scripts\python.exe -m streamlit run streamlit_app/app.py --server.port 8501"

Write-Host "Services started:" -ForegroundColor Green
Write-Host "  FastAPI Backend: http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "  Streamlit UI:    http://localhost:8501" -ForegroundColor Yellow
