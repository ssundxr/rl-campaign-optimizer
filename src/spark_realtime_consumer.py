"""
Spark Streaming Consumer for Real-Time Feature Engineering
Consumes from Kafka, enriches with PostgreSQL data, feeds to LinUCB learner
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import json


class SparkStreamingPipeline:
    """
    Real-time data processing pipeline using Spark Structured Streaming
    
    Flow:
    Kafka (raw events) → Spark (feature engineering) → Kafka (enriched) → LinUCB
    """
    
    def __init__(self, 
                 kafka_bootstrap_servers='localhost:9092',
                 postgres_jdbc_url='jdbc:postgresql://localhost:5432/campaign_analytics',
                 checkpoint_dir='./checkpoints'):
        
        self.kafka_servers = kafka_bootstrap_servers
        self.postgres_url = postgres_jdbc_url
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize Spark with Kafka and PostgreSQL support
        self.spark = SparkSession.builder \
            .appName("RealTimeCampaignOptimizer") \
            .config("spark.jars.packages", 
                   "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                   "org.postgresql:postgresql:42.7.0") \
            .config("spark.sql.streaming.checkpointLocation", checkpoint_dir) \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # Define schema for incoming events
        self.event_schema = StructType([
            StructField("customer_id", IntegerType(), False),
            StructField("event_type", StringType(), False),
            StructField("timestamp", TimestampType(), False),
            StructField("product_id", IntegerType(), True),
            StructField("purchase_amount", DoubleType(), True),
            StructField("session_id", StringType(), True)
        ])
    
    def read_kafka_stream(self, topic='customer-events'):
        """
        Read streaming data from Kafka
        """
        return self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_servers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .load() \
            .selectExpr("CAST(value AS STRING) as json_str", "timestamp as kafka_timestamp")
    
    def parse_events(self, kafka_df):
        """
        Parse JSON events from Kafka
        """
        return kafka_df \
            .select(
                from_json(col("json_str"), self.event_schema).alias("event"),
                col("kafka_timestamp")
            ) \
            .select("event.*", "kafka_timestamp")
    
    def enrich_with_customer_history(self, events_df):
        """
        Join with PostgreSQL to get customer historical data
        """
        # Load customer master data (batch join with streaming)
        customer_master = self.spark.read \
            .format("jdbc") \
            .option("url", self.postgres_url) \
            .option("dbtable", "(SELECT customer_id, total_purchases, total_spent, avg_order_value, days_since_first_purchase FROM customer_summary) as customers") \
            .option("user", "postgres") \
            .option("password", "password") \
            .load()
        
        # Join streaming events with static customer data
        enriched = events_df.join(
            broadcast(customer_master),
            on="customer_id",
            how="left"
        )
        
        return enriched
    
    def compute_realtime_features(self, enriched_df):
        """
        Feature engineering: compute RFM and derived metrics
        """
        features_df = enriched_df.withColumn(
            "features",
            struct(
                # Recency (computed from event timestamp)
                datediff(current_timestamp(), col("timestamp")).alias("recency"),
                
                # Frequency (from master data)
                col("total_purchases").alias("frequency"),
                
                # Monetary (from master data)
                col("total_spent").alias("monetary"),
                
                # Derived features
                (col("total_spent") / col("total_purchases")).alias("avg_purchase"),
                (col("total_spent") * 0.3).alias("estimated_margin"),
                (col("total_spent") * 1.2).alias("projected_clv"),
                
                # Time-based features
                col("days_since_first_purchase").alias("customer_age"),
                (lit(365) / col("total_purchases")).alias("purchase_frequency_days"),
                
                # Segmentation flags
                when(col("total_spent") > 50000, 1).otherwise(0).alias("is_high_value"),
                when(datediff(current_timestamp(), col("timestamp")) > 90, 1).otherwise(0).alias("is_at_risk"),
                when(col("total_purchases") > 20, 1).otherwise(0).alias("is_frequent_buyer"),
                
                # Event-specific
                col("purchase_amount").alias("current_purchase"),
                col("event_type"),
                col("customer_id")
            )
        )
        
        return features_df
    
    def aggregate_windowed_metrics(self, features_df):
        """
        Compute sliding window aggregations for real-time patterns
        """
        windowed = features_df \
            .withWatermark("timestamp", "10 minutes") \
            .groupBy(
                window(col("timestamp"), "5 minutes", "1 minute"),
                col("customer_id")
            ) \
            .agg(
                count("*").alias("event_count_5min"),
                sum("purchase_amount").alias("total_spent_5min"),
                avg("purchase_amount").alias("avg_purchase_5min")
            )
        
        return windowed
    
    def prepare_linucb_context(self, features_df):
        """
        Format features into 21-dimensional vector for LinUCB
        """
        linucb_ready = features_df.select(
            col("customer_id"),
            col("timestamp"),
            array(
                # Core RFM
                col("features.recency").cast("double"),
                col("features.frequency").cast("double"),
                col("features.monetary").cast("double"),
                col("features.avg_purchase").cast("double"),
                col("features.estimated_margin").cast("double"),
                col("features.projected_clv").cast("double"),
                
                # Time features
                col("features.customer_age").cast("double"),
                lit(365.0),  # days in year
                col("features.purchase_frequency_days").cast("double"),
                
                # Placeholder features (would be enriched from more sources)
                lit(5.0), lit(0.3), lit(0.3), lit(0.3), lit(0.25),
                lit(0.0), lit(0.0),
                
                # Segmentation
                col("features.is_high_value").cast("double"),
                col("features.is_at_risk").cast("double"),
                col("features.is_frequent_buyer").cast("double"),
                
                # Recent activity
                when(col("features.recency") <= 30, 1).otherwise(0).cast("double"),
                col("features.projected_clv").cast("double")
            ).alias("context_vector")
        )
        
        return linucb_ready
    
    def write_to_kafka(self, processed_df, output_topic='customer-interactions'):
        """
        Write processed features back to Kafka for LinUCB consumer
        """
        query = processed_df \
            .selectExpr(
                "CAST(customer_id AS STRING) as key",
                "to_json(struct(*)) AS value"
            ) \
            .writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_servers) \
            .option("topic", output_topic) \
            .option("checkpointLocation", f"{self.checkpoint_dir}/kafka_sink") \
            .outputMode("append") \
            .start()
        
        return query
    
    def write_to_console(self, processed_df):
        """
        Debug: write to console
        """
        query = processed_df \
            .writeStream \
            .format("console") \
            .option("truncate", False) \
            .outputMode("append") \
            .start()
        
        return query
    
    def run_pipeline(self):
        """
        Execute complete streaming pipeline
        """
        print("🚀 Starting Spark Structured Streaming Pipeline...")
        
        # Step 1: Read from Kafka
        print("📡 Reading from Kafka topic: customer-events")
        kafka_stream = self.read_kafka_stream()
        
        # Step 2: Parse JSON
        print("🔍 Parsing event data...")
        events = self.parse_events(kafka_stream)
        
        # Step 3: Enrich with PostgreSQL data
        print("🗄️  Enriching with customer history from PostgreSQL...")
        enriched = self.enrich_with_customer_history(events)
        
        # Step 4: Feature engineering
        print("⚙️  Computing real-time features...")
        features = self.compute_realtime_features(enriched)
        
        # Step 5: Prepare LinUCB context
        print("🎯 Formatting for LinUCB consumption...")
        linucb_context = self.prepare_linucb_context(features)
        
        # Step 6: Write to output Kafka topic
        print("📤 Writing to Kafka topic: customer-interactions")
        query = self.write_to_kafka(linucb_context)
        
        # Also write to console for monitoring
        console_query = self.write_to_console(
            linucb_context.select("customer_id", "timestamp", "context_vector")
        )
        
        print("✅ Pipeline running! Press Ctrl+C to stop\n")
        
        # Wait for termination
        query.awaitTermination()


def create_sample_customer_table(spark, postgres_url):
    """
    Create sample customer summary table in PostgreSQL
    (Run this once to set up the database)
    """
    print("📊 Creating sample customer summary table...")
    
    # Generate sample data
    sample_data = [
        (i, 
         int(np.random.uniform(5, 50)),  # total_purchases
         float(np.random.uniform(10000, 100000)),  # total_spent
         float(np.random.uniform(500, 5000)),  # avg_order_value
         int(np.random.uniform(30, 365))  # days_since_first_purchase
        )
        for i in range(1, 10001)
    ]
    
    df = spark.createDataFrame(
        sample_data,
        ["customer_id", "total_purchases", "total_spent", "avg_order_value", "days_since_first_purchase"]
    )
    
    df.write \
        .format("jdbc") \
        .option("url", postgres_url) \
        .option("dbtable", "customer_summary") \
        .option("user", "postgres") \
        .option("password", "password") \
        .mode("overwrite") \
        .save()
    
    print("✅ Sample data created in PostgreSQL!")


if __name__ == '__main__':
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Spark Streaming Pipeline')
    parser.add_argument('--mode', choices=['run', 'setup'], default='run',
                       help='Mode: run (start streaming) or setup (create sample data)')
    
    args = parser.parse_args()
    
    pipeline = SparkStreamingPipeline()
    
    if args.mode == 'setup':
        create_sample_customer_table(
            pipeline.spark,
            pipeline.postgres_url
        )
    else:
        pipeline.run_pipeline()
