"""
Simplified Spark job that runs inside Docker Spark container.
This avoids the need to install Java locally.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import json

def main():
    print("🚀 Starting Spark Streaming Pipeline...")
    
    # Create Spark session (will use Java from Docker container)
    spark = SparkSession.builder \
        .appName("CampaignFeatureEngineering") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.postgresql:postgresql:42.6.0") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("✅ Spark session created successfully")
    
    # Schema for customer events
    event_schema = StructType([
        StructField("customer_id", IntegerType()),
        StructField("event_type", StringType()),
        StructField("timestamp", TimestampType()),
        StructField("campaign_id", IntegerType()),
        StructField("channel", StringType()),
        StructField("revenue", DoubleType())
    ])
    
    # Read from Kafka
    print("📥 Connecting to Kafka topic: customer-events")
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("subscribe", "customer-events") \
        .option("startingOffsets", "latest") \
        .load()
    
    # Parse JSON from Kafka value
    parsed_df = df.select(
        from_json(col("value").cast("string"), event_schema).alias("data")
    ).select("data.*")
    
    # Add processing timestamp
    enriched_df = parsed_df.withColumn("processed_at", current_timestamp())
    
    # Basic feature engineering (simplified for Docker environment)
    featured_df = enriched_df.select(
        col("customer_id"),
        col("event_type"),
        col("timestamp"),
        col("campaign_id"),
        col("channel"),
        col("revenue"),
        col("processed_at"),
        # Create a simple feature vector as JSON string
        to_json(struct(
            col("customer_id"),
            col("revenue"),
            hour("timestamp").alias("hour_of_day"),
            dayofweek("timestamp").alias("day_of_week")
        )).alias("features")
    )
    
    # Write to console for monitoring
    console_query = featured_df.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .option("numRows", 5) \
        .start()
    
    # Write back to Kafka (customer-interactions topic)
    kafka_output = featured_df.select(
        col("customer_id").cast("string").alias("key"),
        to_json(struct("*")).alias("value")
    )
    
    kafka_query = kafka_output.writeStream \
        .outputMode("append") \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("topic", "customer-interactions") \
        .option("checkpointLocation", "/tmp/spark-checkpoint") \
        .start()
    
    print("✅ Streaming pipeline started successfully")
    print("📊 Writing to console and customer-interactions topic")
    print("⏳ Press Ctrl+C to stop...")
    
    # Wait for termination
    kafka_query.awaitTermination()

if __name__ == "__main__":
    main()
