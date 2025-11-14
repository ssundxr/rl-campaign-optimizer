"""
Real-Time Model Retraining Pipeline
Connects Kafka → PostgreSQL → LinUCB for continuous online learning
"""

import json
import numpy as np
import pickle
import time
from datetime import datetime
from kafka import KafkaConsumer
from kafka import KafkaProducer
import psycopg2
from psycopg2.extras import execute_values
import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.linucb_agent import LinUCBAgent


class RealtimeLearningPipeline:
    """
    Online learning pipeline that continuously updates LinUCB model
    with real-time customer feedback from Kafka stream
    """
    
    def __init__(self, 
                 kafka_bootstrap_servers='localhost:9092',
                 postgres_config=None,
                 model_path='models/linucb_trained.pkl',
                 checkpoint_interval=100,
                 save_interval=1000):
        
        self.kafka_servers = kafka_bootstrap_servers
        self.model_path = model_path
        self.checkpoint_interval = checkpoint_interval
        self.save_interval = save_interval
        
        # PostgreSQL configuration
        self.pg_config = postgres_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'campaign_analytics',
            'user': 'postgres',
            'password': 'password'
        }
        
        # Initialize components
        self.agent = None
        self.consumer = None
        self.producer = None
        self.pg_conn = None
        self.update_count = 0
        
        # Metrics
        self.total_rewards = []
        self.running_avg_reward = 0.0
        self.start_time = datetime.now()
        
    def initialize(self):
        """Initialize Kafka, PostgreSQL connections and load model"""
        print("[INIT] Initializing Real-Time Learning Pipeline...")
        
        # Load existing model or create new one
        if os.path.exists(self.model_path):
            print(f"[LOAD] Loading existing model from {self.model_path}")
            self.agent = LinUCBAgent.load_model(self.model_path)
        else:
            print("🆕 Creating new LinUCB agent")
            self.agent = LinUCBAgent(
                n_actions=4,
                context_dim=21,
                alpha=1.0,
                campaign_names=['20% Discount', 'Free Shipping', 'Early Access', 'No Campaign']
            )
        
        # Initialize Kafka consumer
        print("📡 Connecting to Kafka...")
        self.consumer = KafkaConsumer(
            'customer-interactions',
            bootstrap_servers=self.kafka_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='realtime-learner'
        )
        
        # Initialize Kafka producer for predictions
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Initialize PostgreSQL connection
        print("🗄️  Connecting to PostgreSQL...")
        self.pg_conn = psycopg2.connect(**self.pg_config)
        self._create_tables()
        
        print("✅ Pipeline initialized successfully!\n")
        
    def _create_tables(self):
        """Create necessary tables in PostgreSQL"""
        with self.pg_conn.cursor() as cur:
            # Table for real-time interactions
            cur.execute("""
                CREATE TABLE IF NOT EXISTS realtime_interactions (
                    id SERIAL PRIMARY KEY,
                    customer_id INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    context JSONB NOT NULL,
                    recommended_action INTEGER NOT NULL,
                    actual_reward FLOAT NOT NULL,
                    model_version VARCHAR(50),
                    processed BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Table for model performance metrics
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    update_count INTEGER NOT NULL,
                    avg_reward FLOAT NOT NULL,
                    total_interactions INTEGER NOT NULL,
                    model_version VARCHAR(50)
                )
            """)
            
            # Index for faster queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON realtime_interactions(timestamp DESC)
            """)
            
            self.pg_conn.commit()
    
    def process_interaction(self, message):
        """
        Process a single customer interaction from Kafka
        
        Expected message format:
        {
            "customer_id": 12345,
            "timestamp": "2025-11-09T10:30:00",
            "features": [f1, f2, ..., f21],  # 21-dimensional context
            "action_taken": 1,  # Which campaign was shown
            "reward": 2500.0    # Customer response (purchase amount or 0)
        }
        """
        try:
            customer_id = message['customer_id']
            features = np.array(message['features'])
            action = message['action_taken']
            reward = message['reward']
            timestamp = message.get('timestamp', datetime.now().isoformat())
            
            # Update LinUCB model with observed reward
            self.agent.update(features, action, reward)
            self.update_count += 1
            
            # Track metrics
            self.total_rewards.append(reward)
            self.running_avg_reward = np.mean(self.total_rewards[-1000:])  # Last 1000
            
            # Store in PostgreSQL
            self._store_interaction(customer_id, timestamp, features, action, reward)
            
            # Periodic checkpointing
            if self.update_count % self.checkpoint_interval == 0:
                self._log_metrics()
            
            # Periodic model save
            if self.update_count % self.save_interval == 0:
                self._save_checkpoint()
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing interaction: {e}")
            return False
    
    def _store_interaction(self, customer_id, timestamp, features, action, reward):
        """Store interaction in PostgreSQL for audit and analysis"""
        try:
            with self.pg_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO realtime_interactions 
                    (customer_id, timestamp, context, recommended_action, actual_reward, model_version)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    customer_id,
                    timestamp,
                    json.dumps(features.tolist()),
                    action,
                    reward,
                    f"v{self.update_count}"
                ))
                self.pg_conn.commit()
        except Exception as e:
            print(f"⚠️  Database error: {e}")
            self.pg_conn.rollback()
    
    def _log_metrics(self):
        """Log performance metrics"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        throughput = self.update_count / elapsed if elapsed > 0 else 0
        
        print(f"\n📊 Performance Metrics (Update #{self.update_count}):")
        print(f"   ⏱️  Runtime: {elapsed:.1f}s")
        print(f"   ⚡ Throughput: {throughput:.2f} updates/sec")
        print(f"   💰 Avg Reward (last 1000): ₹{self.running_avg_reward:,.2f}")
        print(f"   📈 Total Interactions: {len(self.total_rewards)}\n")
        
        # Store metrics in database
        try:
            with self.pg_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO model_metrics 
                    (timestamp, update_count, avg_reward, total_interactions, model_version)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    datetime.now(),
                    self.update_count,
                    self.running_avg_reward,
                    len(self.total_rewards),
                    f"v{self.update_count}"
                ))
                self.pg_conn.commit()
        except Exception as e:
            print(f"⚠️  Metrics logging error: {e}")
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = self.model_path.replace('.pkl', f'_checkpoint_{self.update_count}.pkl')
        self.agent.save_model(checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Also update main model
        self.agent.save_model(self.model_path)
    
    def predict_and_publish(self, customer_features):
        """
        Make a prediction and publish to Kafka for serving layer
        
        Args:
            customer_features: 21-dim numpy array
            
        Returns:
            dict with prediction results
        """
        action = self.agent.select_action(customer_features)
        expected_rewards = self.agent.get_expected_rewards(customer_features)
        
        prediction = {
            'timestamp': datetime.now().isoformat(),
            'recommended_action': int(action),
            'campaign_name': self.agent.campaign_names[action],
            'expected_reward': float(expected_rewards[action]),
            'all_expected_rewards': expected_rewards.tolist(),
            'model_version': f"v{self.update_count}"
        }
        
        # Publish to Kafka topic for API/Dashboard consumption
        self.producer.send('campaign-predictions', value=prediction)
        
        return prediction
    
    def run_forever(self):
        """
        Main event loop: consume from Kafka and update model in real-time
        """
        print("🔄 Starting real-time learning loop...")
        print("   Listening for messages on topic: customer-interactions")
        print("   Press Ctrl+C to stop\n")
        
        try:
            for message in self.consumer:
                self.process_interaction(message.value)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping pipeline...")
            self._save_checkpoint()
            self.cleanup()
            print("✅ Pipeline stopped gracefully")
    
    def cleanup(self):
        """Clean up resources"""
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
        if self.pg_conn:
            self.pg_conn.close()


def simulate_realtime_data(producer, n_samples=100, delay=0.5):
    """
    Simulate real-time customer interactions for testing
    
    Args:
        producer: Kafka producer instance
        n_samples: Number of interactions to simulate
        delay: Seconds between messages
    """
    print(f"📤 Simulating {n_samples} customer interactions...")
    
    for i in range(n_samples):
        # Generate random customer features
        features = np.random.randn(21)
        features[0] = np.random.randint(1, 50)  # frequency
        features[1] = np.random.uniform(5000, 50000)  # monetary
        features[-1] = np.random.uniform(10000, 100000)  # CLV
        
        # Simulate campaign assignment and reward
        action = np.random.randint(0, 4)
        # Reward based on campaign effectiveness (free shipping performs best)
        base_rewards = [2000, 3500, 1500, 100]  # per campaign
        reward = base_rewards[action] + np.random.normal(0, 500)
        reward = max(0, reward)  # No negative rewards
        
        message = {
            'customer_id': 10000 + i,
            'timestamp': datetime.now().isoformat(),
            'features': features.tolist(),
            'action_taken': action,
            'reward': reward
        }
        
        producer.send('customer-interactions', value=message)
        
        if (i + 1) % 10 == 0:
            print(f"   Sent {i+1}/{n_samples} interactions")
        
        time.sleep(delay)
    
    print("✅ Simulation complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Learning Pipeline')
    parser.add_argument('--mode', choices=['learn', 'simulate'], default='learn',
                       help='Mode: learn (consume and update) or simulate (generate test data)')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples to simulate')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between simulated messages (seconds)')
    
    args = parser.parse_args()
    
    if args.mode == 'learn':
        # Start real-time learning pipeline
        pipeline = RealtimeLearningPipeline()
        pipeline.initialize()
        pipeline.run_forever()
        
    elif args.mode == 'simulate':
        # Simulate customer interactions
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        simulate_realtime_data(producer, n_samples=args.samples, delay=args.delay)
        producer.close()
