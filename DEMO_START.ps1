# ============================================
# CONFERENCE DEMO - ONE-CLICK LAUNCHER
# ============================================

$Host.UI.RawUI.WindowTitle = "Real-Time Learning Demo"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   CONFERENCE DEMO STARTING..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

# Step 1: Check Docker
Write-Host "[Step 1/4] Checking Docker services..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "  Docker is running" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Docker is not running!" -ForegroundColor Red
    Write-Host "  Please start Docker Desktop first." -ForegroundColor Yellow
    pause
    exit 1
}

# Step 2: Start Real-Time Learner (Background)
Write-Host ""
Write-Host "[Step 2/4] Starting Real-Time Learner..." -ForegroundColor Yellow
$env:PYTHONIOENCODING = "utf-8"

$learnerJob = Start-Process python -ArgumentList "src/realtime_learner.py", "--mode", "learn" `
    -PassThru -WindowStyle Minimized -RedirectStandardOutput "logs/learner.log" -RedirectStandardError "logs/learner_err.log"

Write-Host "  Learner started (PID: $($learnerJob.Id))" -ForegroundColor Green
Start-Sleep -Seconds 3

# Step 3: Start Data Simulator (Background)
Write-Host ""
Write-Host "[Step 3/4] Starting Data Simulator..." -ForegroundColor Yellow

$simulatorJob = Start-Process python -ArgumentList "src/realtime_learner.py", "--mode", "simulate", "--samples", "1000", "--delay", "0.5" `
    -PassThru -WindowStyle Minimized -RedirectStandardOutput "logs/simulator.log"

Write-Host "  Simulator started (1000 interactions)" -ForegroundColor Green
Start-Sleep -Seconds 2

# Step 4: Launch Live Dashboard
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "   LAUNCHING LIVE DASHBOARD..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "  Opening browser in 3 seconds..." -ForegroundColor Cyan
Start-Sleep -Seconds 3

# Open browser
Start-Process "http://localhost:8502"

# Start Streamlit dashboard
Write-Host "  Starting Streamlit server..." -ForegroundColor Cyan
streamlit run dashboard/realtime_monitor.py --server.port 8502 --server.headless true

# Cleanup on exit
Write-Host ""
Write-Host "Stopping background processes..." -ForegroundColor Yellow
Stop-Process -Id $learnerJob.Id -ErrorAction SilentlyContinue
Stop-Process -Id $simulatorJob.Id -ErrorAction SilentlyContinue
Write-Host "Demo stopped." -ForegroundColor Green
