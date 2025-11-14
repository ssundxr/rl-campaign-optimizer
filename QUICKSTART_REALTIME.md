# Real-Time Learning Pipeline - Quick Start

## ✅ Prerequisites Complete
- ✅ Kafka topics created (customer-events, customer-interactions, campaign-predictions)
- ✅ Docker services running (Kafka, PostgreSQL, Spark, Zookeeper)

## 🚀 Two Options to Test

### Option 1: Simple Test (Without Spark) ⭐ **RECOMMENDED**

This bypasses Spark and directly tests the LinUCB learner with simulated data.

**Terminal 1** - Start the learner:
```powershell
python src/realtime_learner.py --mode learn
```

**Terminal 2** - Run simulation:
```powershell
python src/realtime_learner.py --mode simulate --samples 100 --delay 0.5
```

Watch Terminal 1 for real-time metrics:
- ⚡ Throughput (events/sec)
- 💰 Average reward
- 📊 Model updates

---

### Option 2: Full Pipeline (With Spark in Docker)

If you want to test the complete architecture including Spark Structured Streaming:

**Terminal 1** - Start Spark streaming job:
```powershell
.\run_spark_streaming.ps1
```

**Terminal 2** - Start LinUCB learner:
```powershell
python src/realtime_learner.py --mode learn
```

**Terminal 3** - Generate events:
```powershell
python src/realtime_learner.py --mode simulate --samples 100 --delay 0.5
```

---

## 📊 Monitor the Pipeline

### Check PostgreSQL metrics:
```powershell
docker exec -it postgres-db psql -U postgres -d campaign_analytics -c "SELECT * FROM model_metrics ORDER BY timestamp DESC LIMIT 10;"
```

### View Kafka messages:
```powershell
docker exec kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic customer-interactions --max-messages 5
```

### Inspect model state:
```powershell
docker exec -it postgres-db psql -U postgres -d campaign_analytics -c "SELECT * FROM realtime_interactions ORDER BY timestamp DESC LIMIT 5;"
```

---

## 🎯 Expected Output

**Learner Terminal:**
```
🚀 Real-Time Learning Pipeline Starting...
✅ Kafka consumer connected
✅ PostgreSQL connected
✅ Model loaded: models/linucb_trained.pkl
📊 Starting to consume from: customer-interactions

[Metrics @ 2025-11-09 15:30:45]
⚡ Throughput: 45.2 events/sec
💰 Avg Reward: $2,847.32
📈 Total Interactions: 452
✅ Model checkpoint saved (every 100 updates)
```

**Simulator Terminal:**
```
📤 Simulating 100 customer interactions...
   Sent 10/100 interactions
   Sent 20/100 interactions
   ...
✅ Simulation complete!
```

---

## 🎓 Next Steps

1. **Integrate with Dashboard**: Add Kafka consumer to `dashboard/app.py` to show live predictions
2. **Scale Up**: Increase partitions and add multiple learner instances
3. **A/B Testing**: Compare online learning vs. static model in production
4. **Model Versioning**: Track performance across model checkpoints

---

## 💡 Troubleshooting

**Q: "No module named 'kafka'"**
```powershell
pip install kafka-python psycopg2-binary
```

**Q: Learner not receiving messages**
- Ensure simulator is publishing to `customer-interactions` topic
- Check Kafka consumer group offsets

**Q: PostgreSQL connection error**
- Verify password is `password` (not `postgres`)
- Check Docker container is running: `docker ps | findstr postgres`

---

## 🔥 Performance Tips

- Reduce `--delay` in simulation for higher throughput (e.g., `--delay 0.01`)
- Monitor system resources with `docker stats`
- Use batch updates for production (currently updates every message)
