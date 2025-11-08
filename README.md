# Real-Time RL Campaign Optimization Engine

> **Self-Learning AI System for E-Commerce Customer Retention**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Spark](https://img.shields.io/badge/Apache%20Spark-3.5-orange.svg)](https://spark.apache.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

Production-grade Reinforcement Learning system that optimizes marketing campaigns in real-time using contextual bandits (LinUCB algorithm). Processes 100K+ transactions to maximize customer lifetime value and reduce churn.

## 🎯 Key Features

- **Real-Time Optimization**: Sub-second campaign recommendations via Kafka streaming
- **Self-Learning AI**: Continuous model improvement through reinforcement learning
- **Big Data Pipeline**: Distributed processing with Apache Spark
- **REST API**: Flask-based microservice for model serving
- **Live Dashboard**: Real-time analytics with Streamlit
- **Containerized**: Full Docker orchestration for easy deployment

## 🏗️ Architecture

```
Data Generation → Feature Engineering → Kafka Streaming → RL Model → API/Dashboard
     (CSV)            (Parquet)           (Events)        (LinUCB)   (Flask/Streamlit)
```

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- Python 3.9+
- 8GB RAM (recommended)

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/rl_campaign_optimizer.git
cd rl_campaign_optimizer
pip install -r requirements.txt
```

### 2. Start Infrastructure
```bash
docker-compose up -d
```

**Services:**
- Spark Master UI: http://localhost:8080
- Kafka: localhost:9092
- PostgreSQL: localhost:5432

### 3. Generate & Process Data
```bash
# Generate 100K transactions
python data/generate_data.py

# Engineer ML features
python src/pandas_feature_pipeline.py
```

### 4. Launch Applications
```bash
# Terminal 1: Start API
python api/app.py

# Terminal 2: Start Dashboard
streamlit run dashboard/app.py
```

**Access:**
- API: http://localhost:5000
- Dashboard: http://localhost:8501

## 📊 Business Impact

| Metric | Expected Impact |
|--------|----------------|
| Campaign Conversion Rate | +25-40% |
| Customer Lifetime Value | +30% |
| Churn Reduction | -20-35% |
| Marketing ROI | +45% |

## 🗂️ Project Structure

```
rl_campaign_optimizer/
├── data/
│   ├── generate_data.py      # Synthetic data generator
│   ├── raw/                   # CSV files (100K transactions)
│   └── processed/             # Parquet features (30 features)
├── src/
│   ├── pandas_feature_pipeline.py    # Feature engineering
│   ├── kafka_producer.py             # Event streaming
│   └── spark_streaming_consumer.py   # Real-time processing
├── api/
│   └── app.py                 # Flask REST API
├── dashboard/
│   └── app.py                 # Streamlit analytics dashboard
├── docker-compose.yml         # Infrastructure orchestration
└── requirements.txt           # Python dependencies
```

## 🧠 ML Approach

**Algorithm**: Linear Upper Confidence Bound (LinUCB)
- **Problem Type**: Contextual Multi-Armed Bandit
- **Features**: RFM analysis, behavioral patterns, customer profiles
- **Reward Function**: Conversion + Revenue - Cost
- **Exploration/Exploitation**: UCB-based action selection

## 🔌 API Endpoints

```bash
# Health Check
GET /

# Get Campaign Recommendation
POST /predict
{
  "customer_id": 1234,
  "features": [0.5, 0.3, 0.8, ...]
}

# Submit Feedback
POST /feedback
{
  "customer_id": 1234,
  "campaign_id": 5,
  "reward": 1.5
}
```

## 🐳 Docker Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| Spark | apache/spark:3.5.0 | 8080 | Distributed processing |
| Kafka | confluentinc/cp-kafka:7.5.0 | 9092 | Event streaming |
| PostgreSQL | postgres:15-alpine | 5432 | Analytics storage |
| Zookeeper | confluentinc/cp-zookeeper:7.5.0 | 2181 | Kafka coordination |

## 📈 Sample Results

- **Customers Processed**: 10,000
- **Transactions**: 100,000
- **High-Value Customers**: 173 (1.7%)
- **At-Risk Customers**: 835 (8.3%)
- **Average CLV**: ₹50,771

## 🛠️ Tech Stack

**ML/Data**: PyTorch, Scikit-learn, XGBoost, PySpark  
**Streaming**: Apache Kafka  
**Database**: PostgreSQL  
**API**: Flask, Flask-CORS  
**Frontend**: Streamlit, Plotly  
**Infrastructure**: Docker, Docker Compose  

## 📝 Development

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**  
[GitHub](https://github.com/yourusername) • [LinkedIn](https://linkedin.com/in/yourprofile)

---

⭐ **Star this repo** if you find it useful!
