"""
Apache Spark Feature Engineering Pipeline
Processes raw customer data into ML-ready features
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, datediff, current_date, count, sum as _sum, 
    avg, max as _max, min as _min, stddev, countDistinct
)
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_session():
    """Initialize Spark session with optimized configurations"""
    spark = SparkSession.builder \
        .appName("RL Campaign Feature Pipeline") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    logger.info("Spark session created successfully")
    return spark


def load_raw_data(spark, transactions_path, customers_path):
    """Load raw customer and transaction data from CSV files"""
    logger.info(f"Loading transactions from {transactions_path}")
    df_transactions = spark.read.csv(transactions_path, header=True, inferSchema=True)
    logger.info(f"Loaded {df_transactions.count():,} transactions")
    
    logger.info(f"Loading customers from {customers_path}")
    df_customers = spark.read.csv(customers_path, header=True, inferSchema=True)
    logger.info(f"Loaded {df_customers.count():,} customers")
    
    return df_transactions, df_customers


def engineer_features(df_transactions, df_customers):
    """Create RFM and behavioral features for RL model"""
    logger.info("Engineering features...")
    
    # Calculate RFM (Recency, Frequency, Monetary) features per customer
    logger.info("Calculating RFM features...")
    rfm_features = df_transactions.groupBy("customer_id").agg(
        count("transaction_id").alias("frequency"),
        _sum("amount").alias("monetary_total"),
        avg("amount").alias("monetary_avg"),
        _max("transaction_date").alias("last_transaction_date"),
        _min("transaction_date").alias("first_transaction_date"),
        countDistinct("category").alias("category_diversity"),
        avg(when(col("discount_used"), 1).otherwise(0)).alias("discount_usage_rate")
    )
    
    # Calculate recency (days since last transaction)
    from pyspark.sql.functions import lit, to_date
    current_date_val = lit("2024-12-31")  # Using end date from data generation
    
    rfm_features = rfm_features.withColumn(
        "recency_days",
        datediff(to_date(current_date_val), col("last_transaction_date"))
    )
    
    # Calculate customer lifetime (days as customer)
    rfm_features = rfm_features.withColumn(
        "customer_lifetime_days",
        datediff(col("last_transaction_date"), col("first_transaction_date"))
    )
    
    # Join with customer profiles
    logger.info("Joining with customer profiles...")
    df_features = rfm_features.join(df_customers, on="customer_id", how="left")
    
    # Add derived features
    df_features = df_features.withColumn(
        "avg_days_between_purchases",
        when(col("frequency") > 1, 
             col("customer_lifetime_days") / (col("frequency") - 1))
        .otherwise(None)
    )
    
    df_features = df_features.withColumn(
        "is_high_value",
        when(col("monetary_total") > 50000, 1).otherwise(0)
    )
    
    df_features = df_features.withColumn(
        "is_at_risk",
        when(col("recency_days") > 180, 1).otherwise(0)
    )
    
    logger.info(f"Feature engineering complete. Created {len(df_features.columns)} features")
    
    return df_features


def save_processed_data(df, output_path):
    """Save processed features to Parquet"""
    logger.info(f"Saving processed data to {output_path}")
    df.write.mode("overwrite").parquet(output_path)
    logger.info("Data saved successfully")


def main():
    """Main pipeline execution"""
    logger.info("="*70)
    logger.info("SPARK FEATURE ENGINEERING PIPELINE")
    logger.info("="*70)
    
    spark = create_spark_session()
    
    # Define paths
    transactions_path = "./data/raw/transactions.csv"
    customers_path = "./data/raw/customers.csv"
    output_path = "./data/processed/features.parquet"
    
    # Ensure output directory exists
    os.makedirs("./data/processed", exist_ok=True)
    
    # Execute pipeline
    df_transactions, df_customers = load_raw_data(spark, transactions_path, customers_path)
    df_features = engineer_features(df_transactions, df_customers)
    
    # Show sample
    logger.info("\nSample of engineered features:")
    df_features.select(
        "customer_id", "frequency", "monetary_total", "recency_days",
        "category_diversity", "is_high_value", "is_at_risk"
    ).show(5, truncate=False)
    
    # Save
    save_processed_data(df_features, output_path)
    
    # Summary
    feature_count = df_features.count()
    logger.info(f"\n{'='*70}")
    logger.info(f"PIPELINE COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Customers processed: {feature_count:,}")
    logger.info(f"Features created: {len(df_features.columns)}")
    logger.info(f"Output: {output_path}")
    logger.info(f"{'='*70}")
    
    spark.stop()
    logger.info("Spark session closed")


if __name__ == "__main__":
    main()
