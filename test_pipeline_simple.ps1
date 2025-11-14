# Simple one-command test for real-time pipeline

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  REAL-TIME LEARNING PIPELINE TEST" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "This script will:" -ForegroundColor Yellow
Write-Host "  1. Start the learner in the background"
Write-Host "  2. Wait 3 seconds for initialization"
Write-Host "  3. Run the simulator (100 interactions)"
Write-Host "  4. Show you the results"
Write-Host ""

Write-Host "Press Enter to start..." -ForegroundColor Green
Read-Host

# Start learner in background
Write-Host ""
Write-Host "Starting learner in background..." -ForegroundColor Cyan
$learnerJob = Start-Job -ScriptBlock {
    Set-Location C:\Users\sdshy\CascadeProjects\DATASCIENCE\rl_campaign_optimizer
    python src/realtime_learner.py --mode learn
}

Write-Host "Learner started (Job ID: $($learnerJob.Id))" -ForegroundColor Green

# Wait for initialization
Write-Host "Waiting 3 seconds for initialization..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Run simulator
Write-Host ""
Write-Host "Starting simulator..." -ForegroundColor Cyan
python src/realtime_learner.py --mode simulate --samples 100 --delay 0.5

Write-Host ""
Write-Host "Simulation complete!" -ForegroundColor Green

# Show learner output
Write-Host ""
Write-Host "Learner output (last 30 lines):" -ForegroundColor Yellow
Receive-Job -Job $learnerJob | Select-Object -Last 30

# Stop learner
Write-Host ""
Write-Host "Stopping learner..." -ForegroundColor Yellow
Stop-Job -Job $learnerJob
Remove-Job -Job $learnerJob

Write-Host "Test complete!" -ForegroundColor Green
Write-Host ""

# Check PostgreSQL metrics
Write-Host "Checking PostgreSQL metrics..." -ForegroundColor Cyan
docker exec -it postgres-db psql -U postgres -d campaign_analytics -c "SELECT COUNT(*) as total_interactions FROM realtime_interactions;" 2>$null

Write-Host ""
Write-Host "Check QUICKSTART_REALTIME.md for more details" -ForegroundColor Gray
Write-Host ""
