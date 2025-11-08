"""
Training Script for LinUCB Reinforcement Learning Agent
Trains on processed customer features to optimize campaign selection

This script:
1. Loads processed customer features (30 dimensions)
2. Simulates realistic rewards based on customer segments
3. Trains LinUCB agent using contextual bandit approach
4. Saves trained model for production deployment
"""

import pandas as pd
import numpy as np
import os
from models.linucb_agent import LinUCBAgent


def load_data(features_path: str) -> np.ndarray:
    """
    Load processed customer features from parquet file.
    
    Args:
        features_path: Path to features.parquet file
    
    Returns:
        Numpy array of shape (n_customers, 30) with normalized features
    """
    print("="*70)
    print("LOADING CUSTOMER FEATURES")
    print("="*70)
    
    # Load parquet file
    df = pd.read_parquet(features_path)
    print(f"✅ Loaded {len(df):,} customers")
    
    # Define exact feature columns in correct order (matching actual parquet file)
    # Only numerical features suitable for ML
    feature_columns = [
        'frequency', 'monetary_total', 'monetary_avg', 'monetary_std',
        'monetary_min', 'monetary_max', 'recency_days', 
        'customer_lifetime_days', 'avg_days_between_purchases',
        'category_diversity', 'discount_usage_rate',
        'weekend_transactions', 'weekend_shopper_ratio',
        'email_open_rate', 'complaint_count', 'is_premium_member',
        'is_high_value', 'is_at_risk', 'is_frequent_buyer', 'is_recent_active',
        'clv_score'
    ]
    
    # Verify all columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Extract features in correct order and convert to float
    X = df[feature_columns].astype(float).values
    
    # Handle any NaN values
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"⚠️  Filling {nan_count:,} NaN values with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    print(f"✅ Features matrix shape: {X.shape}")
    print(f"   - Customers: {X.shape[0]:,}")
    print(f"   - Features: {X.shape[1]}")
    
    return X


def simulate_reward(customer_features: np.ndarray, action: int) -> float:
    """
    Simulate realistic reward based on customer segment and campaign action.
    
    Reward Model:
    - High-value customers respond best to Early Access (exclusive perks)
    - At-risk customers need aggressive Discount campaigns
    - Frequent buyers love Free Shipping
    - Other customers often don't need campaigns
    
    Args:
        customer_features: Feature vector of shape (21,)
        action: Campaign action ID (0-3)
    
    Returns:
        Simulated reward (positive = profit, negative = loss)
    """
    # Extract customer segment indicators (adjusted for actual feature indices)
    is_high_value = customer_features[16]   # Index 16: is_high_value
    is_at_risk = customer_features[17]      # Index 17: is_at_risk
    is_frequent = customer_features[18]     # Index 18: is_frequent_buyer
    clv = customer_features[20]             # Index 20: clv_score
    
    # Define campaign costs
    campaign_costs = {
        0: -50,   # 20% Discount: ₹50 cost
        1: -30,   # Free Shipping: ₹30 cost
        2: -20,   # Early Access: ₹20 cost (minimal cost, high value)
        3: 0      # No Campaign: ₹0 cost
    }
    cost = campaign_costs[action]
    
    # Define conversion probabilities by segment and action
    # [20% Discount, Free Shipping, Early Access, No Campaign]
    if is_high_value == 1:
        # High-value customers respond best to exclusivity (Early Access)
        conversion_probs = [0.3, 0.4, 0.9, 0.2]
    elif is_at_risk == 1:
        # At-risk customers need aggressive incentives (Discount)
        conversion_probs = [0.9, 0.6, 0.4, 0.1]
    elif is_frequent == 1:
        # Frequent buyers appreciate convenience (Free Shipping)
        conversion_probs = [0.5, 0.8, 0.6, 0.3]
    else:
        # Other customers often don't need campaigns
        conversion_probs = [0.4, 0.5, 0.3, 0.5]
    
    # Simulate conversion
    converted = np.random.random() < conversion_probs[action]
    
    # Calculate reward
    if converted:
        # Successful conversion: earn 10% of customer's CLV
        revenue = clv * 0.1
        reward = revenue + cost  # Revenue minus campaign cost
    else:
        # No conversion: lose campaign cost plus churn penalty
        churn_penalty = -50
        reward = churn_penalty + cost
    
    return reward


def train_agent(X: np.ndarray, agent: LinUCBAgent) -> dict:
    """
    Train LinUCB agent on customer features with simulated rewards.
    
    Args:
        X: Feature matrix of shape (n_customers, 21)
        agent: Initialized LinUCBAgent
    
    Returns:
        Dictionary with training statistics
    """
    print("\n" + "="*70)
    print("TRAINING LINUCB AGENT")
    print("="*70)
    
    n_customers = X.shape[0]
    
    # Initialize tracking variables
    total_reward = 0.0
    action_counts = [0, 0, 0, 0]
    
    # Training loop
    np.random.seed(42)  # For reproducible results
    
    for i in range(n_customers):
        # Get customer context
        context = X[i]
        
        # Agent selects action using LinUCB
        action = agent.select_action(context)
        
        # Simulate reward from environment
        reward = simulate_reward(context, action)
        
        # Update agent with observed reward
        agent.update(context, action, reward)
        
        # Track statistics
        total_reward += reward
        action_counts[action] += 1
        
        # Print progress every 2000 customers
        if (i + 1) % 2000 == 0:
            avg_reward = total_reward / (i + 1)
            print(f"  Trained on {i+1:>5,}/{n_customers:,} | "
                  f"Avg Reward: ₹{avg_reward:>8,.2f}")
    
    # Calculate final statistics
    avg_reward = total_reward / n_customers
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total Customers Trained: {n_customers:,}")
    print(f"Total Reward: ₹{total_reward:,.2f}")
    print(f"Average Reward per Customer: ₹{avg_reward:,.2f}")
    
    print("\n" + "-"*70)
    print("CAMPAIGN ACTION DISTRIBUTION")
    print("-"*70)
    
    for action, count in enumerate(action_counts):
        campaign_name = agent.campaign_names[action]
        percentage = (count / n_customers) * 100
        print(f"  {campaign_name:15s}: {count:>5,} customers ({percentage:>5.1f}%)")
    
    # Return statistics
    stats = {
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'action_counts': action_counts,
        'n_customers': n_customers
    }
    
    return stats


def save_model(agent: LinUCBAgent, model_path: str):
    """
    Save trained agent to disk.
    
    Args:
        agent: Trained LinUCBAgent
        model_path: Path where model should be saved
    """
    print("\n" + "="*70)
    print("SAVING TRAINED MODEL")
    print("="*70)
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    agent.save_model(model_path)
    
    # Get model file size
    file_size = os.path.getsize(model_path) / 1024  # KB
    print(f"✅ Model size: {file_size:.2f} KB")
    
    # Print agent statistics
    stats = agent.get_stats()
    print(f"✅ Total interactions trained: {stats['total_interactions']:,}")


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("LINUCB AGENT TRAINING PIPELINE")
    print("Reinforcement Learning for Campaign Optimization")
    print("="*70)
    
    # Configuration
    features_path = 'data/processed/features.parquet'
    model_path = 'models/linucb_trained.pkl'
    
    # Step 1: Load data
    X = load_data(features_path)
    
    # Step 2: Initialize agent
    print("\n" + "="*70)
    print("INITIALIZING LINUCB AGENT")
    print("="*70)
    
    agent = LinUCBAgent(
        n_actions=4,        # 4 campaign types
        context_dim=21,     # 21 customer features (actual count from data)
        alpha=1.0           # Exploration parameter
    )
    
    print(f"✅ Agent initialized")
    print(f"   - Actions: {agent.n_actions} campaign types")
    print(f"   - Context Dimension: {agent.context_dim} features")
    print(f"   - Exploration (alpha): {agent.alpha}")
    print(f"\nCampaign Types:")
    for action_id, campaign_name in agent.campaign_names.items():
        print(f"   {action_id}: {campaign_name}")
    
    # Step 3: Train agent
    stats = train_agent(X, agent)
    
    # Step 4: Save model
    save_model(agent, model_path)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETE")
    print("="*70)
    print(f"Model ready for production deployment at: {model_path}")
    print(f"\nTo use the trained model:")
    print(f"  from models.linucb_agent import LinUCBAgent")
    print(f"  agent = LinUCBAgent.load_model('{model_path}')")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
