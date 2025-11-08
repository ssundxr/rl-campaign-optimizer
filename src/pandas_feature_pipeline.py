"""
Pandas Feature Engineering Pipeline (Local Version)
Processes raw customer data into ML-ready features
Alternative to Spark for local development
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_raw_data(transactions_path, customers_path):
    """Load raw customer and transaction data from CSV files"""
    logger.info(f"Loading transactions from {transactions_path}")
    df_transactions = pd.read_csv(transactions_path, parse_dates=['transaction_date'])
    logger.info(f"✅ Loaded {len(df_transactions):,} transactions")
    
    logger.info(f"Loading customers from {customers_path}")
    df_customers = pd.read_csv(customers_path, parse_dates=['signup_date'])
    logger.info(f"✅ Loaded {len(df_customers):,} customers")
    
    return df_transactions, df_customers


def engineer_features(df_transactions, df_customers):
    """Create RFM and behavioral features for RL model"""
    logger.info("\n[Feature Engineering Started]")
    
    # Calculate RFM (Recency, Frequency, Monetary) features per customer
    logger.info("  → Calculating RFM features...")
    rfm_features = df_transactions.groupby('customer_id').agg({
        'transaction_id': 'count',  # Frequency
        'amount': ['sum', 'mean', 'std', 'min', 'max'],  # Monetary
        'transaction_date': ['max', 'min'],  # Recency
        'category': 'nunique',  # Category diversity
        'discount_used': 'mean'  # Discount usage rate
    })
    
    # Flatten column names
    rfm_features.columns = [
        'frequency', 'monetary_total', 'monetary_avg', 'monetary_std',
        'monetary_min', 'monetary_max', 'last_transaction_date', 
        'first_transaction_date', 'category_diversity', 'discount_usage_rate'
    ]
    
    rfm_features.reset_index(inplace=True)
    
    # Calculate recency (days since last transaction)
    current_date = pd.to_datetime('2024-12-31')  # Using end date from data generation
    rfm_features['recency_days'] = (current_date - rfm_features['last_transaction_date']).dt.days
    
    # Calculate customer lifetime (days as customer)
    rfm_features['customer_lifetime_days'] = (
        rfm_features['last_transaction_date'] - rfm_features['first_transaction_date']
    ).dt.days
    
    # Calculate average days between purchases
    rfm_features['avg_days_between_purchases'] = np.where(
        rfm_features['frequency'] > 1,
        rfm_features['customer_lifetime_days'] / (rfm_features['frequency'] - 1),
        np.nan
    )
    
    logger.info("  → Adding behavioral features...")
    
    # Add device and payment preferences
    device_prefs = df_transactions.groupby('customer_id')['device'].agg(lambda x: x.mode()[0])
    payment_prefs = df_transactions.groupby('customer_id')['payment_method'].agg(lambda x: x.mode()[0])
    
    rfm_features = rfm_features.merge(device_prefs.rename('preferred_device'), 
                                      on='customer_id', how='left')
    rfm_features = rfm_features.merge(payment_prefs.rename('preferred_payment'), 
                                      on='customer_id', how='left')
    
    # Weekend shopping behavior
    weekend_transactions = df_transactions[df_transactions['is_weekend']].groupby('customer_id').size()
    rfm_features = rfm_features.merge(weekend_transactions.rename('weekend_transactions'), 
                                      on='customer_id', how='left')
    rfm_features['weekend_transactions'] = rfm_features['weekend_transactions'].fillna(0)
    rfm_features['weekend_shopper_ratio'] = rfm_features['weekend_transactions'] / rfm_features['frequency']
    
    logger.info("  → Joining with customer profiles...")
    
    # Join with customer profiles
    df_features = rfm_features.merge(df_customers, on='customer_id', how='left')
    
    # Add derived classification features
    logger.info("  → Creating classification features...")
    
    df_features['is_high_value'] = (df_features['monetary_total'] > 50000).astype(int)
    df_features['is_at_risk'] = (df_features['recency_days'] > 180).astype(int)
    df_features['is_frequent_buyer'] = (df_features['frequency'] > 10).astype(int)
    df_features['is_recent_active'] = (df_features['recency_days'] <= 30).astype(int)
    
    # Customer lifetime value score (simplified)
    df_features['clv_score'] = (
        df_features['monetary_total'] * 
        (1 - df_features['recency_days'] / 365) * 
        np.log1p(df_features['frequency'])
    )
    
    logger.info(f"✅ Feature engineering complete. Created {len(df_features.columns)} features")
    
    return df_features


def save_processed_data(df, output_path):
    """Save processed features to Parquet"""
    logger.info(f"\nSaving processed data to {output_path}")
    df.to_parquet(output_path, index=False, engine='pyarrow')
    file_size = os.path.getsize(output_path) / 1_000_000
    logger.info(f"✅ Data saved successfully ({file_size:.2f} MB)")


def main():
    """Main pipeline execution"""
    logger.info("="*70)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("="*70)
    
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define paths relative to project root
    transactions_path = os.path.join(project_root, "data", "raw", "transactions.csv")
    customers_path = os.path.join(project_root, "data", "raw", "customers.csv")
    output_path = os.path.join(project_root, "data", "processed", "features.parquet")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Execute pipeline
    df_transactions, df_customers = load_raw_data(transactions_path, customers_path)
    df_features = engineer_features(df_transactions, df_customers)
    
    # Show sample
    logger.info("\n" + "="*70)
    logger.info("SAMPLE OF ENGINEERED FEATURES")
    logger.info("="*70)
    sample_features = [
        'customer_id', 'frequency', 'monetary_total', 'monetary_avg',
        'recency_days', 'category_diversity', 'is_high_value', 
        'is_at_risk', 'clv_score'
    ]
    print(df_features[sample_features].head(10).to_string(index=False))
    
    # Save
    save_processed_data(df_features, output_path)
    
    # Summary statistics
    logger.info("\n" + "="*70)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*70)
    logger.info(f"Customers processed: {len(df_features):,}")
    logger.info(f"Features created: {len(df_features.columns)}")
    logger.info(f"\nKey Statistics:")
    logger.info(f"  High-value customers: {df_features['is_high_value'].sum():,} ({df_features['is_high_value'].mean()*100:.1f}%)")
    logger.info(f"  At-risk customers: {df_features['is_at_risk'].sum():,} ({df_features['is_at_risk'].mean()*100:.1f}%)")
    logger.info(f"  Frequent buyers: {df_features['is_frequent_buyer'].sum():,} ({df_features['is_frequent_buyer'].mean()*100:.1f}%)")
    logger.info(f"  Avg CLV Score: ₹{df_features['clv_score'].mean():,.2f}")
    logger.info(f"\nOutput saved: {output_path}")
    logger.info("="*70)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
