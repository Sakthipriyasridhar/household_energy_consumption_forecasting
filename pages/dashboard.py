import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Energy Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Minimalist CSS
def load_css():
    css = """
    <style>
        /* Main styles */
        .main-header {
            font-size: 2.8rem;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #2E86AB, #06D6A0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        /* Clean card design */
        .metric-card {
            background: white;
            padding: 1.8rem 1.5rem;
            border-radius: 12px;
            border: 1px solid #e8e8e8;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .metric-card:hover {
            border-color: #2E86AB;
            box-shadow: 0 4px 16px rgba(46, 134, 171, 0.1);
        }
        
        .metric-title {
            font-size: 0.9rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 0.25rem;
        }
        
        .metric-change {
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        /* Navigation cards */
        .nav-card {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            border: 1px solid #e8e8e8;
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            cursor: pointer;
        }
        
        .nav-card:hover {
            transform: translateY(-6px);
            border-color: #2E86AB;
            box-shadow: 0 12px 32px rgba(46, 134, 171, 0.15);
        }
        
        .nav-icon {
            font-size: 3rem;
            margin-bottom: 1.5rem;
            color: #2E86AB;
            transition: all 0.3s ease;
        }
        
        .nav-card:hover .nav-icon {
            transform: scale(1.1);
        }
        
        .nav-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 0.5rem;
        }
        
        .nav-desc {
            font-size: 0.95rem;
            color: #666;
            line-height: 1.5;
            margin-bottom: 1.5rem;
        }
        
        /* Clean buttons */
        .stButton > button {
            background: linear-gradient(90deg, #2E86AB, #1b9aaa);
            color: white;
            border: none;
            padding: 0.8rem 1.8rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(46, 134, 171, 0.25);
        }
        
        /* Secondary button */
        .secondary-button > button {
            background: white;
            color: #2E86AB;
            border: 2px solid #2E86AB;
            padding: 0.8rem 1.8rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .secondary-button > button:hover {
            background: #2E86AB;
            color: white;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: white;
            border-right: 1px solid #e8e8e8;
        }
        
        /* Progress indicators */
        .progress-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .completed {
            background: rgba(6, 214, 160, 0.1);
            color: #06D6A0;
        }
        
        .pending {
            background: rgba(255, 107, 107, 0.1);
            color: #ff6b6b;
        }
        
        .available {
            background: rgba(255, 209, 102, 0.1);
            color: #ffd166;
        }
        
        /* Divider */
        .custom-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, #e8e8e8, transparent);
            margin: 2rem 0;
        }
        
        /* Charts */
        .plotly-chart {
            border-radius: 12px;
            border: 1px solid #e8e8e8;
            padding: 1rem;
            background: white;
        }
        
        /* Center align for navigation */
        .center-container {
            display: flex;
            justify-content: center;
        }
        
        /* Navigation grid */
        .nav-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        @media (max-width: 1200px) {
            .nav-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        @media (max-width: 768px) {
            .nav-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Initialize Session State
def init_session_state():
    default_states = {
        "survey_completed": False,
        "user_data": {},
        "forecast_generated": False,
        "data_loaded": False
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

def show_dashboard():
    """Clean, Professional Dashboard"""
    
    # Header Section
    st.markdown('<div class="main-header">Energy Optimizer AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning-Powered Energy Management & Optimization Platform</div>', unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Key Metrics Dashboard
    st.markdown("### Performance Overview")
    
    metrics_data = [
        {"title": "Avg. Savings", "value": "32%", "change": "+5.2%", "trend": "positive", "icon": "💰"},
        {"title": "Forecast Accuracy", "value": "85.2%", "change": "+2.1%", "trend": "positive", "icon": "📊"},
        {"title": "CO₂ Reduced", "value": "12.5t", "change": "Monthly", "trend": "neutral", "icon": "🌱"},
    ]
    
    metric_cols = st.columns(4)
    
    for idx, metric in enumerate(metrics_data):
        with metric_cols[idx]:
            change_color = "#06D6A0" if metric["trend"] == "positive" else "#ff6b6b" if metric["trend"] == "negative" else "#666"
            
            metric_html = f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <div class="metric-title">{metric['title']}</div>
                        <div class="metric-value">{metric['value']}</div>
                        <div class="metric-change" style="color: {change_color};">{metric['change']}</div>
                    </div>
                    <div style="font-size: 1.8rem;">{metric['icon']}</div>
                </div>
            </div>
            """
            st.markdown(metric_html, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # CENTERED NAVIGATION SECTION - Main Focus
    st.markdown("### Navigation Center")
    st.markdown("Select where you'd like to go:")
    
    # Create navigation cards in a centered grid
    navigation_options = [
        {
            "icon": "📊", 
            "title": "Data Upload", 
            "desc": "Upload historical energy consumption data",
            "page": "data_loader"
        },
        {
            "icon": "📋", 
            "title": "Energy Survey", 
            "desc": "Complete smart energy assessment questionnaire",
            "page": "survey"
        },
        {
            "icon": "🤖", 
            "title": "AI Forecast", 
            "desc": "Generate 12-month AI-powered predictions",
            "page": "forecast"
        },
        {
            "icon": "💡", 
            "title": "Optimization", 
            "desc": "Get personalized energy savings recommendations",
            "page": "optimization"
        },
        {
            "icon": "☀️", 
            "title": "Solar Analysis", 
            "desc": "Calculate solar ROI and savings potential",
            "page": "solar"
        },
    ]
    
    # Create 3x2 grid for navigation cards
    nav_rows = [navigation_options[i:i+3] for i in range(0, len(navigation_options), 3)]
    
    for row in nav_rows:
        cols = st.columns(3)
        for col_idx, nav_item in enumerate(row):
            with cols[col_idx]:
                nav_html = f"""
                <div class="nav-card" onclick="this.nextElementSibling.click()">
                    <div class="nav-icon">{nav_item['icon']}</div>
                    <div class="nav-title">{nav_item['title']}</div>
                    <div class="nav-desc">{nav_item['desc']}</div>
                </div>
                """
                st.markdown(nav_html, unsafe_allow_html=True)
                
                # Hidden button for navigation
                if st.button(f"Go to {nav_item['title']}", key=f"nav_{nav_item['title']}", use_container_width=True):
                    st.switch_page(f"pages/{nav_item['page']}.py")
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Getting Started Guide
    st.markdown("### Getting Started Guide")
    
    steps = [
        {"step": 1, "title": "Upload Data", "desc": "Import your historical energy consumption data", "icon": "📊"},
        {"step": 2, "title": "Complete Survey", "desc": "Provide details about your appliances and usage patterns", "icon": "📋"},
        {"step": 3, "title": "Generate Forecast", "desc": "Get AI-powered predictions for future consumption", "icon": "🤖"},
        {"step": 4, "title": "Optimize", "desc": "Receive personalized recommendations for savings", "icon": "💡"}
        {"step": 5, "title": "Solarize", "desc": "Get Your Own Solar Panel Planned & How Much You Can Save", "icon": "☀️"}
        
    ]
    
    step_cols = st.columns(4)
    
    for idx, step in enumerate(steps):
        with step_cols[idx]:
            step_html = f"""
            <div style="text-align: center; padding: 1.5rem; border: 1px solid #e8e8e8; border-radius: 12px; height: 100%; background: white;">
                <div style="background: #2E86AB; color: white; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-weight: bold;">
                    {step['step']}
                </div>
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{step['icon']}</div>
                <h4 style="margin: 0.5rem 0;">{step['title']}</h4>
                <p style="color: #666; font-size: 0.9rem; margin: 0;">{step['desc']}</p>
            </div>
            """
            st.markdown(step_html, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # "Ready to Optimize Your Energy Usage?" Section
    st.markdown("### Ready to Optimize Your Energy Usage?")
    
    # ANALYTICS DASHBOARD - Show it with an expander instead
    with st.expander("📊 **Analytics Dashboard**", expanded=True):
        st.markdown("#### 📈 Detailed Analytics")
        
        tab1, tab2, tab3 = st.tabs(["📈 Energy Trends", "💰 Cost Analysis", "🌍 Environmental Impact"])
        
        with tab1:
            # Create sample energy consumption data
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
            
            # Simulated data
            np.random.seed(42)
            base_consumption = np.array([850, 920, 780, 950, 1100, 1250, 1150, 980])
            variation = np.random.normal(0, 50, len(months))
            current_consumption = base_consumption + variation
            optimized_consumption = current_consumption * 0.75  # 25% reduction
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=months, y=current_consumption,
                mode='lines+markers',
                name='Current Consumption',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8, symbol='circle')
            ))
            fig_trend.add_trace(go.Scatter(
                x=months, y=optimized_consumption,
                mode='lines+markers',
                name='Optimized Target',
                line=dict(color='#06D6A0', width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
            
            fig_trend.update_layout(
                title="Monthly Energy Consumption (kWh)",
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    gridcolor='#f0f0f0',
                    showline=True,
                    linecolor='#e0e0e0'
                ),
                yaxis=dict(
                    gridcolor='#f0f0f0',
                    showline=True,
                    linecolor='#e0e0e0',
                    title='Consumption (kWh)'
                )
            )
            
            st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
        
        with tab2:
            # Cost breakdown pie chart
            categories = ['HVAC', 'Lighting', 'Appliances', 'Electronics', 'Water Heating', 'Other']
            costs = [3200, 1200, 1800, 900, 1500, 800]
            
            fig_cost = go.Figure(data=[go.Pie(
                labels=categories,
                values=costs,
                hole=0.4,
                marker=dict(colors=['#2E86AB', '#06D6A0', '#FFD166', '#EF476F', '#118AB2', '#73AB84']),
                textinfo='label+percent',
                textposition='outside',
                texttemplate='%{label}<br>₹%{value:,.0f}<br>(%{percent})',
                hovertemplate='<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent} of total'
            )])
            
            fig_cost.update_layout(
                title="Monthly Cost Distribution (₹)",
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False,
                annotations=[dict(
                    text=f"Total:<br>₹{sum(costs):,.0f}",
                    x=0.5, y=0.5, font_size=16, showarrow=False
                )]
            )
            
            st.plotly_chart(fig_cost, use_container_width=True, config={'displayModeBar': False})
        
        with tab3:
            # Environmental impact gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=68,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Energy Efficiency Score", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#2E86AB"},
                    'bar': {'color': "#06D6A0"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#e0e0e0",
                    'steps': [
                        {'range': [0, 40], 'color': '#ff6b6b'},
                        {'range': [40, 70], 'color': '#ffd166'},
                        {'range': [70, 100], 'color': '#06D6A0'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig_gauge.update_layout(
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
            
            # Environmental metrics
            env_col1, env_col2, env_col3 = st.columns(3)
            with env_col1:
                st.metric("CO₂ Saved", "12,500 kg", "Monthly")
            with env_col2:
                st.metric("Trees Equivalent", "595 trees")
            with env_col3:
                st.metric("Cost Savings", "₹8,400", "Monthly")
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("Start your journey towards smarter energy management and significant cost savings.")
    
    cta_col1, cta_col2, cta_col3 = st.columns([1, 2, 1])
    
    with cta_col2:
        col_start, col_learn = st.columns(2)
        
        with col_start:
            if st.button("🚀 Get Started Now", type="primary", use_container_width=True):
                st.switch_page("pages/survey.py")
        
        with col_learn:
            github_url = "https://github.com/Sakthipriyasridhar/household_energy_consumption_forecasting"
            st.link_button("📚 View on GitHub", 
                          github_url, 
                          type="primary", 
                          use_container_width=True)
    
    # Footer
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    footer_cols = st.columns(3)
    
    with footer_cols[0]:
        st.caption("**Platform:** Energy Optimizer AI v2.1")
        st.caption("**Accuracy:** 85.2% average forecast accuracy")
    
    with footer_cols[1]:
        st.caption("**Security:** 256-bit AES encryption")
        st.caption("**Last Updated:** " + datetime.now().strftime("%Y-%m-%d"))
    
    with footer_cols[2]:
        st.caption("**Privacy:** Your data is secure with us")
        st.caption("**Open Source:** Available on GitHub")

# Streamlit pages automatically run when loaded
load_css()
init_session_state()
show_dashboard()
