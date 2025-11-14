# Real-Time Learning Pipeline - Setup & Usage Guide

## Architecture Overview

```
Customer Events → Kafka → Spark Streaming → Feature Engineering → 
    → Kafka (enriched) → LinUCB Learner → Model Update → PostgreSQL
```

## Components

1. **Kafka**: Message broker for event streaming
2. **Spark Structured Streaming**: Real-time feature engineering
3. **PostgreSQL**: Customer history storage
4. **LinUCB Agent**: Online learning algorithm

---

## Quick Start

### Step 1: Ensure Infrastructure is Running

```powershell
# Start Docker services
docker-compose up -d

# Verify all services are healthy
docker ps
```

### Step 2: Create Kafka Topics

```powershell
# Create topic for raw customer events
docker exec kafka kafka-topics --create --topic customer-events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Create topic for enriched interactions (input to LinUCB)
docker exec kafka kafka-topics --create --topic customer-interactions --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Create topic for model predictions
docker exec kafka kafka-topics --create --topic campaign-predictions --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Verify topics created
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092
```

### Step 3: Initialize PostgreSQL Tables

```powershell
# Option 1: Using Python script
python src/realtime_learner.py --mode learn

# Option 2: Manual SQL
docker exec -it postgres-db psql -U postgres -d campaign_analytics -c "
CREATE TABLE IF NOT EXISTS customer_summary (
    customer_id INTEGER PRIMARY KEY,
    total_purchases INTEGER NOT NULL,
    total_spent FLOAT NOT NULL,
    avg_order_value FLOAT NOT NULL,
    days_since_first_purchase INTEGER NOT NULL
);
"

# Insert sample data
python -c "
from src.spark_realtime_consumer import create_sample_customer_table, SparkStreamingPipeline
pipeline = SparkStreamingPipeline()
create_sample_customer_table(pipeline.spark, pipeline.postgres_url)
"
```

---

## Running the Real-Time Pipeline

### Terminal 1: Start Spark Streaming (Feature Engineering)

```powershell
# Process events and enrich with customer history
python src/spark_realtime_consumer.py --mode run
```

**What it does:**
- Reads raw events from `customer-events` topic
- Joins with PostgreSQL customer history
- Computes 21-dimensional feature vectors
- Writes enriched data to `customer-interactions` topic

---

### Terminal 2: Start LinUCB Online Learner

```powershell
# Continuous model updates
python src/realtime_learner.py --mode learn
```

**What it does:**
- Consumes from `customer-interactions` topic
- Updates LinUCB model matrices (A, b) with each interaction
- Saves checkpoints every 100 updates
- Persists full model every 1000 updates
- Logs performance metrics to PostgreSQL

---

### Terminal 3: Simulate Customer Events (Testing)

```powershell
# Generate 1000 synthetic events with 0.5s delay
python src/realtime_learner.py --mode simulate --samples 1000 --delay 0.5
```

**What it generates:**
- Random customer profiles (21 features)
- Campaign assignments (4 actions)
- Simulated rewards based on campaign effectiveness

---

## Monitoring the Pipeline

### View Real-Time Metrics

```powershell
# Watch PostgreSQL metrics table
docker exec -it postgres-db psql -U postgres -d campaign_analytics -c "
SELECT timestamp, update_count, avg_reward, total_interactions 
FROM model_metrics 
ORDER BY timestamp DESC 
LIMIT 10;
"
```

### Inspect Kafka Messages

```powershell
# View raw events
docker exec kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic customer-events --from-beginning --max-messages 5

# View enriched interactions
docker exec kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic customer-interactions --from-beginning --max-messages 5

# View model predictions
docker exec kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic campaign-predictions --from-beginning --max-messages 5
```

### Check Model Performance

```python
# Python REPL
from models.linucb_agent import LinUCBAgent
agent = LinUCBAgent.load_model('models/linucb_trained.pkl')

# View model statistics
stats = agent.get_stats()
print(f"Action counts: {stats['action_counts']}")
print(f"Total updates: {stats['total_updates']}")
```

---

## Production Deployment Workflow

### 1. Cold Start (Initial Training)

```powershell
# Train on historical batch data
python train_rl_model.py

# Start real-time learning from trained baseline
python src/realtime_learner.py --mode learn
```

### 2. Continuous Updates

The model updates automatically as new data arrives. No manual intervention needed.

### 3. Model Rollback (if performance degrades)

```powershell
# List checkpoints
Get-ChildItem models/ -Filter "linucb_trained_checkpoint_*.pkl"

# Restore specific checkpoint
Copy-Item models/linucb_trained_checkpoint_5000.pkl models/linucb_trained.pkl -Force

# Restart learner
python src/realtime_learner.py --mode learn
```

---

## Performance Benchmarks

| Metric | Expected Value |
|--------|---------------|
| **Throughput** | 500-1000 events/sec (single instance) |
| **Latency** | <50ms (Kafka → LinUCB update) |
| **Model Update Time** | <5ms per interaction |
| **Checkpoint Save** | <100ms (every 100 updates) |
| **Memory Footprint** | ~200MB (LinUCB agent + buffers) |

---

## Advanced Features

### A/B Testing in Real-Time

```python
# Modify realtime_learner.py to add epsilon-greedy exploration
class RealtimeLearningPipeline:
    def predict_with_exploration(self, features, epsilon=0.1):
        if np.random.random() < epsilon:
            action = np.random.randint(0, 4)  # Random exploration
        else:
            action = self.agent.select_action(features)  # Exploit
        return action
```

### Multi-Armed Bandit Ensemble

```python
# Train separate models per customer segment
class SegmentedLinUCB:
    def __init__(self):
        self.models = {
            'high_value': LinUCBAgent(...),
            'at_risk': LinUCBAgent(...),
            'frequent_buyer': LinUCBAgent(...)
        }
    
    def select_action(self, features, segment):
        return self.models[segment].select_action(features)
```

---

## Troubleshooting

### Issue: High latency in Spark Streaming

**Solution**: Increase Spark executor memory
```powershell
# Edit spark_realtime_consumer.py
.config("spark.executor.memory", "4g") \
.config("spark.driver.memory", "2g")
```

### Issue: Kafka consumer lag

**Solution**: Increase parallelism
```powershell
# Create more partitions
docker exec kafka kafka-topics --alter --topic customer-interactions --partitions 6 --bootstrap-server localhost:9092

# Run multiple learner instances in consumer group
```

### Issue: PostgreSQL connection timeout

**Solution**: Use connection pooling
```python
from psycopg2 import pool
connection_pool = pool.SimpleConnectionPool(1, 20, **pg_config)
```

---

## Integration with Dashboard

The dashboard automatically picks up model updates:

```python
# dashboard/app.py already loads latest model
agent = LinUCBAgent.load_model('models/linucb_trained.pkl')
```

For real-time predictions, consume from Kafka:

```python
from kafka import KafkaConsumer
consumer = KafkaConsumer('campaign-predictions', ...)
for message in consumer:
    prediction = message.value
    # Display in dashboard
```

---

## Next Steps

1. ✅ **Deployed**: Basic real-time learning pipeline
2. 🚧 **In Progress**: Spark Structured Streaming integration
3. 📋 **Planned**: 
   - Distributed training across Spark cluster
   - Model versioning with MLflow
   - Automated A/B test allocation
   - Feature drift detection (Evidently AI)
   - Cloud deployment (Azure Event Hubs + Databricks)

---

## References

- [Kafka Streaming](https://kafka.apache.org/documentation/streams/)
- [Spark Structured Streaming](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
- [LinUCB Paper](https://arxiv.org/abs/1003.0146)
- [Online Learning Best Practices](https://hunch.net/~jl/projects/online_learning/)
