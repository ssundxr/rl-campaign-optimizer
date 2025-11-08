"""
Generate realistic e-commerce customer campaign dataset
100,000 transactions for 10,000 customers
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

print("="*70)
print("CUSTOMER CAMPAIGN DATA GENERATOR")
print("="*70)

# Config
N_TRANSACTIONS = 100_000
N_CUSTOMERS = 10_000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)
DATE_RANGE_DAYS = (END_DATE - START_DATE).days

print(f"\nGenerating {N_TRANSACTIONS:,} transactions for {N_CUSTOMERS:,} customers...")

# Generate transactions
print("\n[1/3] Generating transaction history...")
transaction_data = {
    'transaction_id': range(1, N_TRANSACTIONS + 1),
    'customer_id': np.random.randint(1, N_CUSTOMERS + 1, N_TRANSACTIONS),
    'transaction_date': [START_DATE + timedelta(days=int(np.random.uniform(0, DATE_RANGE_DAYS))) 
                         for _ in range(N_TRANSACTIONS)],
    'amount': np.round(np.random.exponential(2000, N_TRANSACTIONS) + 500, 2),
    'category': np.random.choice(
        ['Electronics', 'Fashion', 'Groceries', 'Travel', 'FinTech', 'Home', 'Beauty'],
        N_TRANSACTIONS,
        p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]
    ),
    'payment_method': np.random.choice(
        ['Credit Card', 'Debit Card', 'E-Wallet', 'Bank Transfer'],
        N_TRANSACTIONS,
        p=[0.40, 0.30, 0.20, 0.10]
    ),
    'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], N_TRANSACTIONS, p=[0.6, 0.3, 0.1]),
    'discount_used': np.random.choice([True, False], N_TRANSACTIONS, p=[0.3, 0.7])
}

df_transactions = pd.DataFrame(transaction_data)

# Add day features
df_transactions['day_of_week'] = df_transactions['transaction_date'].dt.dayofweek
df_transactions['is_weekend'] = df_transactions['day_of_week'].isin([5, 6])
df_transactions['month'] = df_transactions['transaction_date'].dt.month
df_transactions['is_holiday_season'] = df_transactions['month'].isin([11, 12])

print(f"✅ Generated {len(df_transactions):,} transactions")

# Generate customer profiles
print("\n[2/3] Generating customer profiles...")
customer_data = {
    'customer_id': range(1, N_CUSTOMERS + 1),
    'signup_date': [START_DATE - timedelta(days=int(np.random.uniform(0, 730))) 
                    for _ in range(N_CUSTOMERS)],
    'country': np.random.choice(
        ['Japan', 'Singapore', 'India', 'Taiwan', 'Thailand'],
        N_CUSTOMERS,
        p=[0.30, 0.25, 0.25, 0.10, 0.10]
    ),
    'age_group': np.random.choice(['18-25', '26-35', '36-50', '50+'], N_CUSTOMERS, p=[0.25, 0.40, 0.25, 0.10]),
    'email_verified': np.random.choice([True, False], N_CUSTOMERS, p=[0.85, 0.15]),
    'email_open_rate': np.round(np.random.beta(2, 5, N_CUSTOMERS), 3),
    'complaint_count': np.random.poisson(0.5, N_CUSTOMERS),
    'is_premium_member': np.random.choice([True, False], N_CUSTOMERS, p=[0.15, 0.85])
}

df_customers = pd.DataFrame(customer_data)
print(f"✅ Generated {len(df_customers):,} customer profiles")

# Save files
print("\n[3/3] Saving data...")
os.makedirs('data/raw', exist_ok=True)

transactions_path = 'data/raw/transactions.csv'
customers_path = 'data/raw/customers.csv'

df_transactions.to_csv(transactions_path, index=False)
df_customers.to_csv(customers_path, index=False)

print(f"✅ Saved: {transactions_path} ({os.path.getsize(transactions_path)/1_000_000:.2f} MB)")
print(f"✅ Saved: {customers_path} ({os.path.getsize(customers_path)/1_000_000:.2f} MB)")

print("\n" + "="*70)
print("DATA GENERATION COMPLETE")
print("="*70)
print(f"\n📊 SUMMARY:")
print(f"  Transactions: {len(df_transactions):,}")
print(f"  Customers: {len(df_customers):,}")
print(f"  Total GMV: ₹{df_transactions['amount'].sum():,.2f}")
print(f"  Date Range: {df_transactions['transaction_date'].min().date()} to {df_transactions['transaction_date'].max().date()}")
print(f"\n✅ Ready for Spark processing!")
print("="*70)
