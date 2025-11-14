# Script to run Spark Streaming job inside Docker Spark container

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "SPARK STREAMING PIPELINE (DOCKER MODE)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "📦 Copying Spark job to container..." -ForegroundColor Yellow
docker cp src/spark_submit_job.py spark-master:/opt/spark-apps/spark_submit_job.py

Write-Host "🚀 Submitting job to Spark cluster..." -ForegroundColor Yellow
Write-Host "`n⚠️  Note: This will stream continuously. Press Ctrl+C to stop.`n" -ForegroundColor Magenta

docker exec -it spark-master spark-submit `
    --master local[*] `
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.postgresql:postgresql:42.6.0 `
    /opt/spark-apps/spark_submit_job.py
