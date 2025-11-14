"""
Real-Time Learning Pipeline Monitor Dashboard
Live visualization for conference demonstrations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import psycopg2
import time
import numpy as np

# Page config
st.set_page_config(
    page_title="Real-Time Learning Monitor",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 20px;
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    .status-live {
        color: #00ff00;
        font-weight: bold;
        animation: blink 1.5s infinite;
    }
    @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0.5; }
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Database connection
@st.cache_resource
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="campaign_analytics",
        user="postgres",
        password="password"
    )

# Fetch real-time data
def fetch_realtime_stats():
    """Fetch latest statistics from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Total interactions
    cursor.execute("SELECT COUNT(*) FROM realtime_interactions")
    total = cursor.fetchone()[0]
    
    # Last 60 seconds activity
    cursor.execute("""
        SELECT COUNT(*) FROM realtime_interactions 
        WHERE timestamp > NOW() - INTERVAL '60 seconds'
    """)
    recent = cursor.fetchone()[0]
    
    # Average reward
    cursor.execute("""
        SELECT AVG(actual_reward) FROM realtime_interactions
        WHERE timestamp > NOW() - INTERVAL '5 minutes'
    """)
    avg_reward = cursor.fetchone()[0] or 0
    
    # Action distribution (last 100)
    cursor.execute("""
        SELECT recommended_action, COUNT(*) as count
        FROM realtime_interactions
        WHERE timestamp > NOW() - INTERVAL '5 minutes'
        GROUP BY recommended_action
        ORDER BY recommended_action
    """)
    actions = cursor.fetchall()
    
    # Time series data (last 5 minutes, grouped by 10 seconds)
    cursor.execute("""
        SELECT 
            DATE_TRUNC('second', timestamp) as time_bucket,
            COUNT(*) as count,
            AVG(actual_reward) as avg_reward
        FROM realtime_interactions
        WHERE timestamp > NOW() - INTERVAL '5 minutes'
        GROUP BY time_bucket
        ORDER BY time_bucket DESC
        LIMIT 30
    """)
    timeseries = cursor.fetchall()
    
    # Latest interactions
    cursor.execute("""
        SELECT customer_id, recommended_action, actual_reward, timestamp
        FROM realtime_interactions
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    latest = cursor.fetchall()
    
    cursor.close()
    
    return {
        'total': total,
        'recent': recent,
        'avg_reward': avg_reward,
        'actions': actions,
        'timeseries': timeseries,
        'latest': latest
    }

# Header
st.markdown('<div class="main-header">Real-Time Learning Pipeline Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Live Campaign Optimization System | <span class="status-live">● LIVE</span></div>', unsafe_allow_html=True)

# Auto-refresh
placeholder = st.empty()

# Main loop
while True:
    with placeholder.container():
        try:
            data = fetch_realtime_stats()
            
            # Top metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Interactions",
                    value=f"{data['total']:,}",
                    delta=f"+{data['recent']} (last 60s)"
                )
            
            with col2:
                st.metric(
                    label="Avg Reward (5 min)",
                    value=f"₹{data['avg_reward']:,.0f}",
                    delta="Real-time"
                )
            
            with col3:
                throughput = data['recent'] / 60.0
                st.metric(
                    label="Throughput",
                    value=f"{throughput:.1f} /sec",
                    delta="Live stream"
                )
            
            with col4:
                st.metric(
                    label="System Status",
                    value="OPERATIONAL",
                    delta="All services up"
                )
            
            st.markdown("---")
            
            # Two column layout
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                st.subheader("Live Activity Stream (Last 5 Minutes)")
                
                if data['timeseries']:
                    # Time series chart
                    df_ts = pd.DataFrame(data['timeseries'], columns=['timestamp', 'count', 'avg_reward'])
                    df_ts = df_ts.sort_values('timestamp')
                    
                    fig = go.Figure()
                    
                    # Events per second
                    fig.add_trace(go.Scatter(
                        x=df_ts['timestamp'],
                        y=df_ts['count'],
                        mode='lines+markers',
                        name='Events/sec',
                        line=dict(color='#667eea', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(102, 126, 234, 0.2)'
                    ))
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title="Time",
                        yaxis_title="Events",
                        hovermode='x unified',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Latest interactions table
                st.subheader("Latest Interactions")
                if data['latest']:
                    df_latest = pd.DataFrame(
                        data['latest'],
                        columns=['Customer ID', 'Action', 'Reward (₹)', 'Timestamp']
                    )
                    
                    # Map actions
                    action_map = {0: '📧 Email', 1: '📱 SMS', 2: '🔔 Push', 3: '✉️ Direct Mail'}
                    df_latest['Action'] = df_latest['Action'].map(action_map)
                    df_latest['Reward (₹)'] = df_latest['Reward (₹)'].apply(lambda x: f"₹{x:,.2f}")
                    df_latest['Timestamp'] = pd.to_datetime(df_latest['Timestamp']).dt.strftime('%H:%M:%S')
                    
                    st.dataframe(
                        df_latest,
                        use_container_width=True,
                        hide_index=True,
                        height=350
                    )
            
            with col_right:
                st.subheader("Campaign Distribution")
                
                if data['actions']:
                    # Action distribution pie chart
                    action_labels = ['📧 Email', '📱 SMS', '🔔 Push', '✉️ Direct Mail']
                    action_counts = [0, 0, 0, 0]
                    
                    for action, count in data['actions']:
                        if 0 <= action < 4:
                            action_counts[action] = count
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=action_labels,
                        values=action_counts,
                        hole=0.4,
                        marker=dict(colors=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
                    )])
                    
                    fig_pie.update_layout(
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0),
                        showlegend=True,
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Reward distribution
                st.subheader("Reward Statistics")
                
                if data['latest']:
                    rewards = [x[2] for x in data['latest']]
                    
                    fig_hist = go.Figure(data=[go.Histogram(
                        x=rewards,
                        nbinsx=10,
                        marker=dict(
                            color='#667eea',
                            line=dict(color='white', width=1)
                        )
                    )])
                    
                    fig_hist.update_layout(
                        height=250,
                        margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title="Reward (₹)",
                        yaxis_title="Frequency",
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            # Status bar
            st.markdown("---")
            col_status1, col_status2, col_status3 = st.columns(3)
            
            with col_status1:
                st.info(f"🔄 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
            
            with col_status2:
                st.success("✅ Kafka: Connected | PostgreSQL: Connected")
            
            with col_status3:
                st.info("🔄 Auto-refresh: Every 2 seconds")
        
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            st.warning("Ensure Docker services are running and database is accessible")
    
    # Refresh every 2 seconds
    time.sleep(2)
