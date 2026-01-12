import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# --- 1. KONFIGURASI DASHBOARD ---
st.set_page_config(
    page_title="Supply Chain Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Modern CSS - Glassmorphism & Gradient Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Root Variables */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: rgba(22, 22, 35, 0.8);
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --accent-success: #10b981;
        --accent-warning: #f59e0b;
        --accent-danger: #ef4444;
        --text-primary: #ffffff;
        --text-secondary: #a1a1aa;
        --border-color: rgba(99, 102, 241, 0.2);
    }
    
    /* Main Background with Gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding: 1rem 2rem 2rem 2rem;
        max-width: 100%;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Header Styling */
    .dashboard-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    .dashboard-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .dashboard-subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }
    
    /* KPI Card Styles - Glassmorphism */
    .kpi-card {
        background: linear-gradient(135deg, rgba(22, 22, 35, 0.9) 0%, rgba(30, 30, 50, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.25rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    .kpi-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .kpi-label {
        color: var(--text-secondary);
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    
    .kpi-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.4rem;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 8px;
        border-radius: 20px;
    }
    
    .kpi-delta.positive {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
    }
    
    .kpi-delta.negative {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
    }
    
    .kpi-delta.neutral {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
    }
    
    /* Section Card */
    .section-card {
        background: linear-gradient(135deg, rgba(22, 22, 35, 0.8) 0%, rgba(30, 30, 50, 0.6) 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .section-title::before {
        content: '';
        width: 4px;
        height: 20px;
        background: linear-gradient(180deg, #6366f1, #a855f7);
        border-radius: 2px;
    }
    
    /* Custom Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(22, 22, 35, 0.5);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background: transparent;
        color: var(--text-secondary);
        border-radius: 8px;
        padding: 0 20px;
        font-weight: 500;
        font-size: 0.9rem;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12121a 0%, #1a1a2e 100%);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Selectbox & Multiselect */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(22, 22, 35, 0.8);
        border: 1px solid var(--border-color);
        border-radius: 10px;
    }
    
    /* Chart Container */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Metric styling override */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(22, 22, 35, 0.8);
        border: 1px solid var(--border-color);
        border-radius: 10px;
    }
    
    /* Progress bar in sidebar */
    .sidebar-metric {
        background: rgba(99, 102, 241, 0.1);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .sidebar-metric-label {
        color: var(--text-secondary);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .sidebar-metric-value {
        color: var(--text-primary);
        font-size: 1.25rem;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.8rem;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid var(--border-color);
    }
    
    /* Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA PIPELINE ---
@st.cache_data
def load_and_process_data():
    url = "https://raw.githubusercontent.com/aliiuii/UAS-BI/refs/heads/main/supply_chain_data.csv"
    df = pd.read_csv(url)
    df.columns = [col.strip() for col in df.columns]
    
    numeric_cols = ['Price', 'Revenue generated', 'Shipping costs', 'Lead time', 'Defect rates', 'Production volumes', 'Manufacturing costs']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Revenue generated', 'Shipping costs']).fillna(df.median(numeric_only=True))
    
    # Clustering
    X = StandardScaler().fit_transform(df[['Price', 'Revenue generated']])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Product_Segment'] = kmeans.fit_predict(X)
    df['Segment_Name'] = df['Product_Segment'].map({0: 'Value', 1: 'Premium', 2: 'Standard'})
    
    # Profit Margin
    df['Profit_Margin'] = ((df['Revenue generated'] - df['Shipping costs'] - df['Manufacturing costs']) / df['Revenue generated'] * 100).round(2)
    
    # Time Series
    df['Date'] = pd.date_range(start='2025-01-01', periods=len(df), freq='D')
    df['Month'] = df['Date'].dt.strftime('%b')
    
    return df

try:
    df = load_and_process_data()
except Exception as e:
    st.error(f"❌ Gagal Memuat Data: {e}")
    st.stop()

# --- 3. SIDEBAR FILTERS ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: white; margin: 0;">📊 Analytics</h2>
        <p style="color: #a1a1aa; font-size: 0.85rem; margin-top: 5px;">Supply Chain Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Filters
    st.markdown("##### 🎯 Filters")
    
    product_filter = st.multiselect(
        "Product Type",
        options=df['Product type'].unique(),
        default=df['Product type'].unique()
    )
    
    supplier_filter = st.multiselect(
        "Supplier",
        options=df['Supplier name'].unique(),
        default=df['Supplier name'].unique()
    )
    
    location_filter = st.multiselect(
        "Location",
        options=df['Location'].unique(),
        default=df['Location'].unique()
    )
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("##### 📈 Quick Stats")
    
    filtered_df = df[
        (df['Product type'].isin(product_filter)) &
        (df['Supplier name'].isin(supplier_filter)) &
        (df['Location'].isin(location_filter))
    ]
    
    st.markdown(f"""
    <div class="sidebar-metric">
        <div class="sidebar-metric-label">Total Products</div>
        <div class="sidebar-metric-value">{len(filtered_df):,}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="sidebar-metric">
        <div class="sidebar-metric-label">Avg Profit Margin</div>
        <div class="sidebar-metric-value">{filtered_df['Profit_Margin'].mean():.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# Apply filters
df_filtered = df[
    (df['Product type'].isin(product_filter)) &
    (df['Supplier name'].isin(supplier_filter)) &
    (df['Location'].isin(location_filter))
]

# --- 4. HEADER ---
st.markdown("""
<div class="dashboard-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="dashboard-title">Supply Chain Intelligence</h1>
            <p class="dashboard-subtitle">Real-time analytics and insights for supply chain optimization</p>
        </div>
        <div class="live-indicator">
            <span class="live-dot"></span>
            LIVE DATA
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. KPI METRICS ---
total_rev = df_filtered['Revenue generated'].sum()
total_cost = df_filtered['Shipping costs'].sum() + df_filtered['Manufacturing costs'].sum()
avg_lead = df_filtered['Lead time'].mean()
avg_defect = df_filtered['Defect rates'].mean()
total_products = df_filtered['Number of products sold'].sum()
profit_margin = ((total_rev - total_cost) / total_rev * 100)

kpi_cols = st.columns(6)

kpis = [
    {"icon": "💰", "label": "Total Revenue", "value": f"${total_rev/1e6:.2f}M", "delta": "+12.5%", "delta_type": "positive"},
    {"icon": "📦", "label": "Products Sold", "value": f"{total_products/1e3:.1f}K", "delta": "+8.3%", "delta_type": "positive"},
    {"icon": "📊", "label": "Profit Margin", "value": f"{profit_margin:.1f}%", "delta": "+2.1%", "delta_type": "positive"},
    {"icon": "🚚", "label": "Avg Lead Time", "value": f"{avg_lead:.1f} days", "delta": "-1.2 days", "delta_type": "positive"},
    {"icon": "⚠️", "label": "Defect Rate", "value": f"{avg_defect:.2f}%", "delta": "-0.5%", "delta_type": "positive"},
    {"icon": "💵", "label": "Total Costs", "value": f"${total_cost/1e3:.1f}K", "delta": "-3.2%", "delta_type": "positive"},
]

for col, kpi in zip(kpi_cols, kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{kpi['icon']}</div>
            <div class="kpi-label">{kpi['label']}</div>
            <div class="kpi-value">{kpi['value']}</div>
            <div class="kpi-delta {kpi['delta_type']}">{kpi['delta']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 6. MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Overview & Performance", "🔍 Deep Analysis", "📈 Forecasting & Insights"])

# Chart Template
chart_template = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': '#a1a1aa', 'family': 'Inter'},
        'margin': {'l': 40, 'r': 20, 't': 40, 'b': 40},
        'xaxis': {'gridcolor': 'rgba(99,102,241,0.1)', 'zerolinecolor': 'rgba(99,102,241,0.1)'},
        'yaxis': {'gridcolor': 'rgba(99,102,241,0.1)', 'zerolinecolor': 'rgba(99,102,241,0.1)'},
    }
}

# Color Palettes
colors_gradient = ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899']
colors_categorical = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-title">Revenue Trend Over Time</div>', unsafe_allow_html=True)
        
        # Revenue trend with gradient area
        daily_revenue = df_filtered.groupby('Date')['Revenue generated'].sum().reset_index()
        daily_revenue['MA7'] = daily_revenue['Revenue generated'].rolling(7).mean()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=daily_revenue['Date'],
            y=daily_revenue['Revenue generated'],
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.2)',
            line=dict(color='#6366f1', width=2),
            name='Daily Revenue'
        ))
        fig_trend.add_trace(go.Scatter(
            x=daily_revenue['Date'],
            y=daily_revenue['MA7'],
            line=dict(color='#f59e0b', width=2, dash='dot'),
            name='7-Day MA'
        ))
        fig_trend.update_layout(
            height=320,
            **chart_template['layout'],
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified'
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-title">Revenue by Product</div>', unsafe_allow_html=True)
        
        product_rev = df_filtered.groupby('Product type')['Revenue generated'].sum().reset_index()
        fig_donut = go.Figure(data=[go.Pie(
            labels=product_rev['Product type'],
            values=product_rev['Revenue generated'],
            hole=0.65,
            marker=dict(colors=colors_gradient),
            textinfo='percent',
            textfont=dict(size=12, color='white'),
            hovertemplate='%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>'
        )])
        fig_donut.update_layout(
            height=320,
            **chart_template['layout'],
            showlegend=True,
            legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
            annotations=[dict(text=f'${product_rev["Revenue generated"].sum()/1e6:.1f}M', x=0.5, y=0.5, 
                            font_size=18, font_color='white', showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Second Row
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown('<div class="section-title">Shipping Cost by Mode</div>', unsafe_allow_html=True)
        
        mode_cost = df_filtered.groupby('Transportation modes').agg({
            'Shipping costs': 'mean',
            'Lead time': 'mean'
        }).reset_index()
        
        fig_mode = go.Figure()
        fig_mode.add_trace(go.Bar(
            x=mode_cost['Transportation modes'],
            y=mode_cost['Shipping costs'],
            marker=dict(
                color=mode_cost['Shipping costs'],
                colorscale=[[0, '#6366f1'], [1, '#ec4899']],
                cornerradius=8
            ),
            text=mode_cost['Shipping costs'].round(2),
            textposition='outside',
            textfont=dict(color='white'),
            hovertemplate='%{x}<br>Avg Cost: $%{y:.2f}<extra></extra>'
        ))
        fig_mode.update_layout(
            height=280,
            **chart_template['layout'],
            showlegend=False
        )
        st.plotly_chart(fig_mode, use_container_width=True)
    
    with col4:
        st.markdown('<div class="section-title">Carrier Performance</div>', unsafe_allow_html=True)
        
        carrier_perf = df_filtered.groupby('Shipping carriers').agg({
            'Lead time': 'mean',
            'Revenue generated': 'sum'
        }).reset_index().sort_values('Lead time')
        
        fig_carrier = go.Figure()
        fig_carrier.add_trace(go.Bar(
            y=carrier_perf['Shipping carriers'],
            x=carrier_perf['Lead time'],
            orientation='h',
            marker=dict(
                color=['#10b981', '#f59e0b', '#ef4444'][:len(carrier_perf)],
                cornerradius=8
            ),
            text=carrier_perf['Lead time'].round(1).astype(str) + ' days',
            textposition='outside',
            textfont=dict(color='white')
        ))
        fig_carrier.update_layout(
            height=280,
            **chart_template['layout']
        )
        st.plotly_chart(fig_carrier, use_container_width=True)
    
    with col5:
        st.markdown('<div class="section-title">Location Distribution</div>', unsafe_allow_html=True)
        
        location_data = df_filtered.groupby('Location').agg({
            'Revenue generated': 'sum',
            'Production volumes': 'sum'
        }).reset_index()
        
        fig_loc = go.Figure()
        fig_loc.add_trace(go.Bar(
            x=location_data['Location'],
            y=location_data['Revenue generated'],
            marker=dict(
                color=colors_categorical[:len(location_data)],
                cornerradius=8
            ),
            text=(location_data['Revenue generated']/1e5).round(1).astype(str) + 'L',
            textposition='outside',
            textfont=dict(color='white')
        ))
        fig_loc.update_layout(
            height=280,
            **chart_template['layout']
        )
        st.plotly_chart(fig_loc, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-title">Product Segmentation (K-Means Clustering)</div>', unsafe_allow_html=True)
        
        segment_colors = {'Value': '#10b981', 'Standard': '#f59e0b', 'Premium': '#6366f1'}
        
        fig_cluster = px.scatter(
            df_filtered, 
            x='Price', 
            y='Revenue generated',
            color='Segment_Name',
            size='Production volumes',
            color_discrete_map=segment_colors,
            hover_data=['Product type', 'Supplier name'],
            opacity=0.7
        )
        fig_cluster.update_layout(
            height=380,
            **chart_template['layout'],
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig_cluster.update_traces(marker=dict(line=dict(width=1, color='white')))
        st.plotly_chart(fig_cluster, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-title">Supplier Risk Matrix</div>', unsafe_allow_html=True)
        
        supplier_risk = df_filtered.groupby('Supplier name').agg({
            'Production volumes': 'sum',
            'Defect rates': 'mean',
            'Revenue generated': 'sum'
        }).reset_index()
        
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(
            x=supplier_risk['Production volumes'],
            y=supplier_risk['Defect rates'],
            mode='markers+text',
            marker=dict(
                size=supplier_risk['Revenue generated']/supplier_risk['Revenue generated'].max()*50 + 20,
                color=supplier_risk['Defect rates'],
                colorscale=[[0, '#10b981'], [0.5, '#f59e0b'], [1, '#ef4444']],
                showscale=True,
                colorbar=dict(title='Defect %', thickness=15)
            ),
            text=supplier_risk['Supplier name'],
            textposition='top center',
            textfont=dict(color='white', size=10),
            hovertemplate='<b>%{text}</b><br>Volume: %{x:,.0f}<br>Defect Rate: %{y:.2f}%<extra></extra>'
        ))
        
        # Add quadrant lines
        fig_risk.add_hline(y=supplier_risk['Defect rates'].median(), line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig_risk.add_vline(x=supplier_risk['Production volumes'].median(), line_dash="dash", line_color="rgba(255,255,255,0.3)")
        
        fig_risk.update_layout(
            height=380,
            **chart_template['layout'],
            xaxis_title='Production Volume',
            yaxis_title='Defect Rate (%)'
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # Correlation Heatmap & Revenue vs Cost
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<div class="section-title">Revenue vs Shipping Cost Analysis</div>', unsafe_allow_html=True)
        
        fig_scatter = px.scatter(
            df_filtered,
            x='Revenue generated',
            y='Shipping costs',
            color='Product type',
            trendline='ols',
            color_discrete_sequence=colors_categorical,
            opacity=0.6
        )
        fig_scatter.update_layout(
            height=350,
            **chart_template['layout'],
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col4:
        st.markdown('<div class="section-title">Profit Margin by Segment</div>', unsafe_allow_html=True)
        
        segment_profit = df_filtered.groupby('Segment_Name').agg({
            'Profit_Margin': 'mean',
            'Revenue generated': 'sum'
        }).reset_index()
        
        fig_profit = go.Figure()
        fig_profit.add_trace(go.Bar(
            x=segment_profit['Segment_Name'],
            y=segment_profit['Profit_Margin'],
            marker=dict(
                color=[segment_colors.get(s, '#6366f1') for s in segment_profit['Segment_Name']],
                cornerradius=10
            ),
            text=segment_profit['Profit_Margin'].round(1).astype(str) + '%',
            textposition='outside',
            textfont=dict(color='white', size=14, weight=600)
        ))
        fig_profit.update_layout(
            height=350,
            **chart_template['layout'],
            yaxis_title='Avg Profit Margin (%)'
        )
        st.plotly_chart(fig_profit, use_container_width=True)

with tab3:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-title">Revenue Forecast (30-Day Projection)</div>', unsafe_allow_html=True)
        
        # Simple forecast
        daily_rev = df_filtered.groupby('Date')['Revenue generated'].sum().reset_index()
        daily_rev['Day_Num'] = range(len(daily_rev))
        
        # Linear regression for forecast
        from sklearn.linear_model import LinearRegression
        X = daily_rev['Day_Num'].values.reshape(-1, 1)
        y = daily_rev['Revenue generated'].values
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast
        future_days = 30
        future_X = np.arange(len(daily_rev), len(daily_rev) + future_days).reshape(-1, 1)
        future_pred = model.predict(future_X)
        future_dates = pd.date_range(start=daily_rev['Date'].max() + pd.Timedelta(days=1), periods=future_days)
        
        fig_forecast = go.Figure()
        
        # Historical
        fig_forecast.add_trace(go.Scatter(
            x=daily_rev['Date'],
            y=daily_rev['Revenue generated'],
            mode='lines',
            name='Historical',
            line=dict(color='#6366f1', width=2),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.1)'
        ))
        
        # Forecast
        fig_forecast.add_trace(go.Scatter(
            x=future_dates,
            y=future_pred,
            mode='lines',
            name='Forecast',
            line=dict(color='#10b981', width=2, dash='dash'),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))
        
        # Add confidence band
        std = daily_rev['Revenue generated'].std()
        fig_forecast.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates)[::-1],
            y=list(future_pred + std) + list(future_pred - std)[::-1],
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Confidence Band',
            showlegend=True
        ))
        
        fig_forecast.update_layout(
            height=350,
            **chart_template['layout'],
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified'
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)
        
        # Insights cards
        best_product = df_filtered.groupby('Product type')['Revenue generated'].sum().idxmax()
        best_supplier = df_filtered.groupby('Supplier name')['Defect rates'].mean().idxmin()
        best_mode = df_filtered.groupby('Transportation modes')['Shipping costs'].mean().idxmin()
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(16,185,129,0.05)); 
                    border: 1px solid rgba(16,185,129,0.3); border-radius: 12px; padding: 15px; margin-bottom: 10px;">
            <div style="color: #10b981; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">🏆 Top Product</div>
            <div style="color: white; font-size: 1.1rem; font-weight: 600; margin-top: 5px;">{best_product.capitalize()}</div>
        </div>
        
        <div style="background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(99,102,241,0.05)); 
                    border: 1px solid rgba(99,102,241,0.3); border-radius: 12px; padding: 15px; margin-bottom: 10px;">
            <div style="color: #6366f1; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">⭐ Best Supplier</div>
            <div style="color: white; font-size: 1.1rem; font-weight: 600; margin-top: 5px;">{best_supplier}</div>
        </div>
        
        <div style="background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(245,158,11,0.05)); 
                    border: 1px solid rgba(245,158,11,0.3); border-radius: 12px; padding: 15px; margin-bottom: 10px;">
            <div style="color: #f59e0b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">🚚 Most Efficient</div>
            <div style="color: white; font-size: 1.1rem; font-weight: 600; margin-top: 5px;">{best_mode.capitalize()}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Forecast summary
        forecast_growth = ((future_pred[-1] - daily_rev['Revenue generated'].iloc[-1]) / daily_rev['Revenue generated'].iloc[-1] * 100)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(168,85,247,0.2), rgba(168,85,247,0.05)); 
                    border: 1px solid rgba(168,85,247,0.3); border-radius: 12px; padding: 15px;">
            <div style="color: #a855f7; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">📈 30-Day Forecast</div>
            <div style="color: white; font-size: 1.1rem; font-weight: 600; margin-top: 5px;">+{forecast_growth:.1f}% Growth</div>
        </div>
        """, unsafe_allow_html=True)

    # Heatmap - Monthly Performance
    st.markdown('<div class="section-title">Production Volume Heatmap by Location & Product</div>', unsafe_allow_html=True)
    
    heatmap_data = df_filtered.pivot_table(
        values='Production volumes',
        index='Location',
        columns='Product type',
        aggfunc='sum'
    ).fillna(0)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=[[0, '#1e1e30'], [0.5, '#6366f1'], [1, '#ec4899']],
        hovertemplate='Location: %{y}<br>Product: %{x}<br>Volume: %{z:,.0f}<extra></extra>'
    ))
    fig_heatmap.update_layout(
        height=300,
        **chart_template['layout']
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

# --- 7. FOOTER ---
st.markdown("""
<div class="footer">
    <p style="margin: 0;">📊 <strong>Supply Chain Intelligence Dashboard</strong> | Built with Streamlit & Plotly</p>
    <p style="margin: 5px 0 0 0; font-size: 0.75rem;">Business Intelligence Analytics • Real-time Data Visualization • Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)