# ==============================================================================
# AUTOSAR HLD Document Analysis Assistant - Docker Runner
# ==============================================================================
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   AUTOSAR HLD Document Analysis Assistant - Docker Deployment   " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Check Docker availability
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Docker is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Docker Desktop and start the Docker daemon." -ForegroundColor Yellow
    exit 1
}

# 2. Build and start containers
Write-Host "`n[1/4] Starting Docker Compose services (Ollama, Backend, Frontend)..." -ForegroundColor Green
docker compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to start Docker Compose services." -ForegroundColor Red
    exit 1
}

# 3. Wait for services to become healthy
Write-Host "`n[2/4] Waiting for services to become healthy..." -ForegroundColor Green
$maxAttempts = 30
$attempt = 0
$backendReady = $false

while (($attempt -lt $maxAttempts) -and (-not $backendReady)) {
    Start-Sleep -Seconds 2
    $attempt++
    try {
        $res = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 3 -ErrorAction SilentlyContinue
        if ($res.status -eq "healthy") {
            $backendReady = $true
        }
    } catch {
        Write-Host "  ... waiting for backend (attempt $attempt/$maxAttempts)" -ForegroundColor Gray
    }
}

if (-not $backendReady) {
    Write-Host "[WARNING] Backend is taking longer than expected. Check 'docker compose logs backend'." -ForegroundColor Yellow
} else {
    Write-Host "  [OK] Backend API is healthy!" -ForegroundColor Green
}

# 4. Ensure Ollama model is available
Write-Host "`n[3/4] Ensuring Ollama model (llama3.2:3b) is downloaded in container volume..." -ForegroundColor Green
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($health.ollama.model_downloaded -ne $true) {
        Write-Host "  Pulling llama3.2:3b inside autosar_ollama container (this may take a few minutes on first run)..." -ForegroundColor Yellow
        docker compose exec ollama ollama pull llama3.2:3b
    } else {
        Write-Host "  [OK] llama3.2:3b model is already cached in volume." -ForegroundColor Green
    }
} catch {
    Write-Host "  Note: You can pull the model manually using: docker compose exec ollama ollama pull llama3.2:3b" -ForegroundColor Gray
}

# 5. Summary & URLs
Write-Host "`n[4/4] Deployment Summary & Access URLs:" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  Streamlit Web UI:   http://localhost:8501" -ForegroundColor Yellow
Write-Host "  FastAPI Swagger:    http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "  System Health:      http://localhost:8000/health" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  To view logs:       docker compose logs -f" -ForegroundColor White
Write-Host "  To run test suite:  docker compose exec backend pytest -s tests/" -ForegroundColor White
Write-Host "  To stop services:   docker compose down" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor Cyan
