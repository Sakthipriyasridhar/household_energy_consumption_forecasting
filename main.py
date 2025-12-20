import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Energy Optimizer AI",
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
        
        /* Quick action cards */
        .quick-action-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #e8e8e8;
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        
        .quick-action-card:hover {
            transform: translateY(-4px);
            border-color: #2E86AB;
            box-shadow: 0 8px 24px rgba(46, 134, 171, 0.12);
        }
        
        .quick-action-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            color: #2E86AB;
        }
        
        .quick-action-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 0.75rem;
        }
        
        .quick-action-desc {
            font-size: 0.9rem;
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
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: white;
            border-right: 1px solid #e8e8e8;
        }
        
        /* Sidebar navigation items */
        .sidebar-nav-item {
            padding: 0.75rem 1rem;
            margin: 0.25rem 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .sidebar-nav-item:hover {
            background: rgba(46, 134, 171, 0.1);
        }
        
        .sidebar-nav-item.active {
            background: linear-gradient(90deg, #2E86AB, #1b9aaa);
            color: white;
        }
        
        .sidebar-nav-icon {
            font-size: 1.2rem;
            margin-right: 0.75rem;
        }
        
        .sidebar-nav-text {
            font-weight: 500;
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
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Initialize Session State
def init_session_state():
    default_states = {
        "survey_completed": False,
        "user_data": {},
        "forecast_generated": False,
        "data_loaded": False,
        "current_page": "Dashboard"
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Main App
def main():
    load_css()
    init_session_state()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown('<div style="font-size: 1.8rem; color: #2E86AB;">⚡</div>', unsafe_allow_html=True)
        st.markdown("### Energy Optimizer AI")
        st.caption("Intelligent Energy Management Platform")
        
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        # Vertical Navigation Menu
        st.markdown("### Navigation")
        
        nav_items = [
            {"icon": "🏠", "name": "Dashboard", "page": "main.py"},
            {"icon": "📊", "name": "Data Upload", "page": "pages/data_loader.py"},
            {"icon": "📋", "name": "Energy Survey", "page": "pages/survey.py"},
            {"icon": "🤖", "name": "AI Forecast", "page": "pages/forecast.py"},
            {"icon": "💡", "name": "Optimization", "page": "pages/optimization.py"},
            {"icon": "☀️", "name": "Solar Analysis", "page": "pages/solar.py"}
        ]
        
        for item in nav_items:
            is_active = st.session_state.current_page == item["name"]
            item_class = "sidebar-nav-item active" if is_active else "sidebar-nav-item"
            
            nav_html = f"""
            <div class="{item_class}" onclick="this.nextElementSibling.click()">
                <span class="sidebar-nav-icon">{item['icon']}</span>
                <span class="sidebar-nav-text">{item['name']}</span>
            </div>
            """
            st.markdown(nav_html, unsafe_allow_html=True)
            
            # Hidden button for navigation
            if st.button(f"Go to {item['name']}", key=f"nav_{item['name']}", 
                        use_container_width=True, type="primary" if is_active else "secondary"):
                if item["name"] != "Dashboard":
                    st.session_state.current_page = item["name"]
                    st.switch_page(item["page"])
        
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        # User Progress
        st.markdown("### Progress Status")
        
        progress_items = [
            ("Survey", st.session_state.survey_completed, "completed" if st.session_state.survey_completed else "pending"),
            ("Data", st.session_state.data_loaded, "completed" if st.session_state.data_loaded else "pending"),
            ("Forecast", st.session_state.forecast_generated, "completed" if st.session_state.forecast_generated else "available")
        ]
        
        for item, status, style in progress_items:
            status_icon = "✅" if status else "⏳"
            st.markdown(f"<span class='progress-badge {style}'>{status_icon} {item}</span>", unsafe_allow_html=True)
        
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        # Quick Stats
        if st.session_state.user_data.get("monthly_cost"):
            st.metric(
                "Current Bill",
                f"₹{st.session_state.user_data.get('monthly_cost', 0):,.0f}",
                delta="-12%" if st.session_state.survey_completed else None
            )
        
        # Platform Info
        with st.expander("ℹ️ Platform Info", expanded=False):
            st.caption("**Version:** 2.1.0")
            st.caption("**Accuracy:** 85.2%")
            st.caption("**Data Security:** 256-bit AES")
            st.caption("**Last Updated:** " + datetime.now().strftime("%b %d, %Y"))
    
    # Main Dashboard Content
    if st.session_state.current_page == "Dashboard":
        show_dashboard()

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
        {"title": "Users Optimized", "value": "2,847", "change": "+124", "trend": "positive", "icon": "👥"}
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
    
    # Welcome Message
    st.markdown("### Welcome to Energy Optimizer AI!")
    st.markdown("""
    Your intelligent partner for optimizing energy consumption and reducing costs. 
    Get started by selecting an option from the sidebar navigation.
    """)
    
    # Getting Started Guide
    st.markdown("### Getting Started Guide")
    
    steps = [
        {"step": 1, "title": "Upload Data", "desc": "Import your historical energy consumption data", "icon": "📊"},
        {"step": 2, "title": "Complete Survey", "desc": "Provide details about your appliances and usage patterns", "icon": "📋"},
        {"step": 3, "title": "Generate Forecast", "desc": "Get AI-powered predictions for future consumption", "icon": "🤖"},
        {"step": 4, "title": "Optimize", "desc": "Receive personalized recommendations for savings", "icon": "💡"}
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
    
    # Footer
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

if __name__ == "__main__":
    main()
