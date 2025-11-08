"""
Spark Streaming Consumer with RL Model Training
Processes real-time Kafka events and updates LinUCB model
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_streaming_session():
    """Initialize Spark session for streaming"""
    spark = SparkSession.builder \
        .appName("RL Campaign Streaming Consumer") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .getOrCreate()
    
    logger.info("Spark Streaming session created")
    return spark


def define_event_schema():
    """Define schema for incoming Kafka events"""
    schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("event_type", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("campaign_id", IntegerType(), True),
        StructField("value", DoubleType(), True)
    ])
    return schema


def process_stream(spark, kafka_bootstrap_servers='localhost:9092', topic='customer_events'):
    """Process streaming data from Kafka"""
    logger.info(f"Starting to consume from Kafka topic: {topic}")
    
    # Read from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
        .option("subscribe", topic) \
        .option("startingOffsets", "latest") \
        .load()
    
    # Parse JSON events
    schema = define_event_schema()
    events_df = df.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")
    
    # Process events (add RL model update logic here)
    
    # Write to console for monitoring
    query = events_df.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", "false") \
        .start()
    
    logger.info("Streaming query started")
    query.awaitTermination()


def main():
    """Main execution"""
    spark = create_spark_streaming_session()
    
    try:
        process_stream(spark)
    except KeyboardInterrupt:
        logger.info("Streaming stopped by user")
    finally:
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    main()
