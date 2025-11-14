@echo off
REM ============================================
REM CONFERENCE DEMO - ONE-CLICK LAUNCHER
REM ============================================

color 0A
echo.
echo ========================================
echo    STARTING DEMO IN 3 STEPS...
echo ========================================
echo.

REM Step 1: Check Docker
echo [Step 1/3] Checking Docker services...
docker ps >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop first.
    pause
    exit /b 1
)
echo  Docker is running

REM Step 2: Start Real-Time Learner (Background)
echo.
echo [Step 2/3] Starting Real-Time Learner...
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
start /MIN "LinUCB-Learner" cmd /c "python src/realtime_learner.py --mode learn"
echo  Learner started in background
timeout /t 3 /nobreak >nul

REM Step 3: Start Data Simulator (Background)
echo.
echo [Step 3/3] Starting Data Simulator...
start /MIN "Data-Simulator" cmd /c "python src/realtime_learner.py --mode simulate --samples 1000 --delay 0.5"
echo  Simulator started (1000 interactions)
timeout /t 2 /nobreak >nul

REM Step 4: Launch Live Dashboard
echo.
echo ========================================
echo    LAUNCHING LIVE DASHBOARD...
echo ========================================
echo.
echo  Opening browser in 3 seconds...
timeout /t 3 /nobreak >nul

REM Open dashboard
start "" "http://localhost:8502"
streamlit run dashboard/realtime_monitor.py --server.port 8502 --server.headless true

REM Keep window open
echo.
echo ========================================
echo    DEMO IS LIVE!
echo ========================================
echo.
echo  Dashboard: http://localhost:8502
echo  Press Ctrl+C to stop
echo.
pause
