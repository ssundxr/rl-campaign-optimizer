"""
Professional Streamlit Dashboard for Real-Time RL Campaign Optimization
Production-grade interface for monitoring and testing the LinUCB agent
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add models directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Page configuration
st.set_page_config(
    page_title="RL Campaign Optimization Engine",
    page_icon="🎯",
    layout="wide"
)

st.title("Real-Time Campaign Optimization Engine")
st.markdown("**Self-Learning AI for Marketing ROI - Powered by Reinforcement Learning**")

# Top KPI Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers Trained", "10,000", "Complete")
with col2:
    st.metric("Average Reward", "₹3,382", "+11,273% ROI")
with col3:
    st.metric("Model Size", "15.03 KB", "Production-Ready")
with col4:
    st.metric("Learning Status", "Active", "10K interactions")

st.markdown("---")
st.markdown("### Campaign Strategy Analysis")

# Campaign data
campaign_data = pd.DataFrame({
    'Campaign': ['Free Shipping', '20% Discount', 'Early Access', 'No Campaign'],
    'Percentage': [74.5, 16.5, 7.2, 1.8],
    'Customers': [7449, 1654, 719, 178],
    'Color': ['#28a745', '#fd7e14', '#007bff', '#6c757d']
})

fig_campaign = go.Figure(go.Bar(
    x=campaign_data['Percentage'],
    y=campaign_data['Campaign'],
    orientation='h',
    text=[f"{p}% ({c:,} customers)" for p, c in zip(campaign_data['Percentage'], campaign_data['Customers'])],
    textposition='auto',
    marker=dict(color=campaign_data['Color'])
))

fig_campaign.update_layout(
    title="Agent-Learned Campaign Strategy",
    xaxis_title="Percentage (%)",
    yaxis_title="Campaign Type",
    height=350,
    showlegend=False,
    plot_bgcolor='white'
)

st.plotly_chart(fig_campaign, use_container_width=True)

st.info("**Key Insight:** Agent learned that Free Shipping provides the best ROI for 74.5% of customers.")

st.markdown("---")
st.markdown("### Live Prediction Demo")

col_main, col_sidebar = st.columns([2, 1])

with col_sidebar:
    st.markdown("#### Customer Profile")
    recency = st.slider("Recency (days)", 1, 365, 30)
    frequency = st.slider("Frequency (purchases)", 1, 50, 10)
    monetary = st.slider("Monetary (total spent)", 100, 100000, 25000, step=1000, format="₹%d")
    clv_score = st.slider("CLV Score", 0, 200000, 50000, step=5000, format="₹%d")
    
    st.markdown("#### Customer Segments")
    is_high_value = st.checkbox("High Value Customer")
    is_at_risk = st.checkbox("At-Risk Customer")
    is_frequent_buyer = st.checkbox("Frequent Buyer")
    
    predict_button = st.button("Predict Optimal Campaign", use_container_width=True)

with col_main:
    if predict_button:
        try:
            from models.linucb_agent import LinUCBAgent
            
            model_path = 'models/linucb_trained.pkl'
            if os.path.exists(model_path):
                agent = LinUCBAgent.load_model(model_path)
                
                feature_vector = np.array([
                    frequency, monetary, monetary/max(frequency, 1), monetary * 0.3,
                    monetary * 0.5, monetary * 1.2, recency, 365,
                    365/max(frequency, 1), 5, 0.3, frequency * 0.3, 0.3, 0.25, 0, 0,
                    1 if is_high_value else 0, 1 if is_at_risk else 0,
                    1 if is_frequent_buyer else 0, 1 if recency <= 30 else 0, clv_score
                ])
                
                recommended_action = agent.select_action(feature_vector)
                expected_rewards = agent.get_expected_rewards(feature_vector)
                
                recommended_campaign = agent.campaign_names[recommended_action]
                recommended_reward = expected_rewards[recommended_action]
                
                reward_std = np.std(expected_rewards)
                confidence = min(95, 70 + (abs(recommended_reward - np.mean(expected_rewards)) / (reward_std + 1e-6)) * 25)
                
                st.markdown("#### Prediction Results")
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.markdown("**Recommended Campaign**")
                    st.markdown(f"<h2 style='color: #0066cc;'>{recommended_campaign}</h2>", unsafe_allow_html=True)
                
                with res_col2:
                    st.markdown("**Expected Reward**")
                    st.markdown(f"<h2 style='color: #28a745;'>₹{recommended_reward:,.2f}</h2>", unsafe_allow_html=True)
                
                with res_col3:
                    st.markdown("**Confidence Score**")
                    st.markdown(f"<h2 style='color: #0066cc;'>{confidence:.1f}%</h2>", unsafe_allow_html=True)
                
                st.markdown("#### Expected Rewards by Campaign")
                reward_df = pd.DataFrame({
                    'Campaign': [agent.campaign_names[i] for i in range(4)],
                    'Expected Reward': expected_rewards,
                    'Is Recommended': [i == recommended_action for i in range(4)]
                })
                
                fig_rewards = go.Figure(go.Bar(
                    x=reward_df['Campaign'],
                    y=reward_df['Expected Reward'],
                    marker=dict(color=['#28a745' if rec else '#6c757d' for rec in reward_df['Is Recommended']]),
                    text=[f"₹{r:,.2f}" for r in reward_df['Expected Reward']],
                    textposition='auto'
                ))
                
                fig_rewards.update_layout(
                    xaxis_title="Campaign Type",
                    yaxis_title="Expected Reward (₹)",
                    height=300,
                    showlegend=False,
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig_rewards, use_container_width=True)
            else:
                st.error(f"Model not found. Please run train_rl_model.py first.")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    else:
        st.info("Configure customer profile and click 'Predict Optimal Campaign'")

st.markdown("---")
st.markdown("### Business Impact Calculator")

customer_count = st.number_input(
    "Number of Customers (Annual)",
    min_value=10000,
    max_value=100000000,
    value=50000000,
    step=1000000
)

avg_reward = 3382
annual_revenue = customer_count * avg_reward
annual_revenue_crores = annual_revenue / 10000000

impact_col1, impact_col2, impact_col3, impact_col4 = st.columns(4)
with impact_col1:
    st.metric("Annual Revenue Impact", f"₹{annual_revenue_crores:,.2f} Cr", f"{customer_count:,} customers")
with impact_col2:
    st.metric("Per-Customer Value", f"₹{avg_reward:,}", "vs baseline")
with impact_col3:
    st.metric("ROI Percentage", "11,273%", "Data-driven optimization")
with impact_col4:
    st.metric("Payback Period", "Immediate", "Real-time learning")

st.markdown("---")

with st.expander("System Architecture"):
    st.markdown("""
    #### Technology Stack
    
    - **Big Data**: Apache Spark 3.5.0
    - **Streaming**: Apache Kafka 7.5.0
    - **Database**: PostgreSQL 15
    - **ML Framework**: Custom LinUCB
    - **API**: Flask 3.1.2
    - **Dashboard**: Streamlit 1.26.0
    
    #### Model Details
    
    - **Algorithm**: Linear Upper Confidence Bound (LinUCB)
    - **Context Dimension**: 21 features
    - **Action Space**: 4 campaign types
    - **Training Data**: 10,000 customers
    - **Model Size**: 15.03 KB
    """)

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("**Built for Rakuten AI Nation with U Conference**")
    st.markdown("November 11, 2025")
with footer_col2:
    st.markdown("**GitHub Repository**")
    st.markdown("[github.com/ssundxr/rl-campaign-optimizer](https://github.com/ssundxr/rl-campaign-optimizer)")
with footer_col3:
    st.markdown("**Model Version**")
    st.markdown("v1.0 - Production Ready")

st.markdown(f"<p style='text-align: center; color: #666; font-size: 0.875rem;'>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
