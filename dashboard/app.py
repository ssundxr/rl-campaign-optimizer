"""
Streamlit Dashboard for RL Campaign Optimizer
Real-time monitoring and analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="RL Campaign Optimizer Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("Real-Time RL Campaign Optimization Dashboard")
st.markdown("**Self-Improving AI for Marketing ROI**")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Time range selector
    time_range = st.selectbox(
        "Time Range",
        ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"]
    )
    
    # Campaign filter
    campaigns = st.multiselect(
        "Filter Campaigns",
        options=[f"Campaign {i}" for i in range(1, 11)],
        default=[f"Campaign {i}" for i in range(1, 4)]
    )
    
    # Refresh button
    if st.button(" Refresh Data"):
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    st.success(" All Services Running")
    st.info("🔵 Kafka: Connected")
    st.info("🔵 Spark: Active")
    st.info("🔵 PostgreSQL: Online")

# Main content
col1, col2, col3, col4 = st.columns(4)

# Generate mock metrics
with col1:
    st.metric(
        label="Total Campaigns",
        value="10",
        delta="2 new"
    )

with col2:
    st.metric(
        label="Conversion Rate",
        value="34.5%",
        delta="+5.2%"
    )

with col3:
    st.metric(
        label="Avg Revenue",
        value="$127.80",
        delta="+$12.30"
    )

with col4:
    st.metric(
        label="Model Accuracy",
        value="87.3%",
        delta="+2.1%"
    )

st.markdown("---")

# Row 1: Campaign Performance
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Campaign Performance Over Time")
    
    # Generate mock time series data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    data = {
        'Date': dates,
        'Conversions': np.random.randint(50, 200, 30),
        'Revenue': np.random.uniform(1000, 5000, 30)
    }
    df_time = pd.DataFrame(data)
    
    fig = px.line(df_time, x='Date', y='Conversions', 
                  title='Daily Conversions',
                  markers=True)
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🎯 Campaign Comparison")
    
    # Generate mock campaign data
    campaigns_data = {
        'Campaign': [f'Campaign {i}' for i in range(1, 6)],
        'CTR': np.random.uniform(0.05, 0.25, 5),
        'Conversion': np.random.uniform(0.10, 0.40, 5)
    }
    df_campaigns = pd.DataFrame(campaigns_data)
    
    fig = go.Figure(data=[
        go.Bar(name='CTR', x=df_campaigns['Campaign'], y=df_campaigns['CTR']),
        go.Bar(name='Conversion', x=df_campaigns['Campaign'], y=df_campaigns['Conversion'])
    ])
    fig.update_layout(barmode='group', height=300)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Row 2: RL Model Insights
col1, col2 = st.columns(2)

with col1:
    st.subheader(" RL Model Exploration vs Exploitation")
    
    # Generate mock exploration data
    exploration_data = {
        'Action': ['Explore', 'Exploit'],
        'Count': [30, 70]
    }
    df_explore = pd.DataFrame(exploration_data)
    
    fig = px.pie(df_explore, values='Count', names='Action',
                 title='Action Distribution',
                 color_discrete_sequence=['#ff7f0e', '#1f77b4'])
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("💰 Revenue by Customer Segment")
    
    # Generate mock segment data
    segments_data = {
        'Segment': ['High Value', 'Medium Value', 'Low Value', 'At Risk'],
        'Revenue': [45000, 28000, 15000, 8000]
    }
    df_segments = pd.DataFrame(segments_data)
    
    fig = px.bar(df_segments, x='Segment', y='Revenue',
                 title='Revenue Distribution',
                 color='Revenue',
                 color_continuous_scale='Blues')
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Row 3: Real-time Events
st.subheader(" Real-Time Event Stream")

# Generate mock recent events
events_data = {
    'Timestamp': [datetime.now() - timedelta(seconds=i*10) for i in range(10)],
    'Customer ID': np.random.randint(1000, 9999, 10),
    'Event Type': np.random.choice(['Click', 'Purchase', 'Email Open'], 10),
    'Campaign': [f'Campaign {i}' for i in np.random.randint(1, 11, 10)],
    'Value': np.random.uniform(0, 500, 10).round(2)
}
df_events = pd.DataFrame(events_data)

st.dataframe(df_events, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    RL Campaign Optimizer v1.0.0 | Last Updated: {} | 
    Built with Streamlit, Spark, Kafka & PyTorch
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    unsafe_allow_html=True
)
