# Simple test script for real-time pipeline without Spark complexity

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "SIMPLIFIED REAL-TIME LEARNING TEST" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "This version skips Spark and directly feeds events to the LinUCB learner.`n" -ForegroundColor Gray

Write-Host "📋 Setup Steps:" -ForegroundColor Yellow
Write-Host "   1. ✅ Kafka topics created (customer-events, customer-interactions, campaign-predictions)" -ForegroundColor Green
Write-Host "   2. 🔄 Starting LinUCB learner..." -ForegroundColor Yellow

Write-Host "`n🎯 In this terminal, we'll start the learner.`n" -ForegroundColor Cyan
Write-Host "⚠️  Open a SECOND terminal and run this command to simulate data:" -ForegroundColor Magenta
Write-Host "   python src/realtime_learner.py --mode simulate --samples 100 --delay 0.5`n" -ForegroundColor White

Write-Host "Press Enter to start the learner..." -ForegroundColor Yellow
Read-Host

Write-Host "`n🚀 Starting Real-Time Learning Pipeline...`n" -ForegroundColor Green

python src/realtime_learner.py --mode learn
