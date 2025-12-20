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
        
        /* Vertical layout for quick actions */
        .vertical-actions-container {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
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

# Main App
def main():
    load_css()
    init_session_state()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown('<div class="main-header" style="font-size: 1.8rem;">⚡</div>', unsafe_allow_html=True)
        st.markdown("### Energy Optimizer AI")
        st.caption("Intelligent Energy Management Platform")
        
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        # Navigation
        nav_options = [
            {"icon": "🏠", "name": "Dashboard", "page": "main.py"},
            {"icon": "📊", "name": "Data Upload", "page": "pages/data_loader.py"},
            {"icon": "📋", "name": "Energy Survey", "page": "pages/survey.py"},
            {"icon": "🤖", "name": "AI Forecast", "page": "pages/forecast.py"},
            {"icon": "💡", "name": "Optimization", "page": "pages/optimization.py"},
            {"icon": "☀️", "name": "Solar Analysis", "page": "pages/solar.py"}
        ]
        
        selected = st.selectbox(
            "Navigate to",
            options=[opt["name"] for opt in nav_options],
            format_func=lambda x: f"{next(opt['icon'] for opt in nav_options if opt['name'] == x)} {x}",
            label_visibility="collapsed"
        )
        
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
    
    # Handle page navigation
    if selected != "Dashboard":
        target_page = next(opt["page"] for opt in nav_options if opt["name"] == selected)
        st.info(f"Redirecting to {selected}...")
        st.switch_page(target_page)
    else:
        # Show Dashboard content
        show_dashboard()
    
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
    
    # Main Content Area - Split layout
    main_col = st.columns()
    
    with main_col:
        # Vertical Quick Actions Section
        st.markdown("### Quick Actions")
        st.markdown("<div class='vertical-actions-container'>", unsafe_allow_html=True)
        
        quick_actions = [
            {"icon": "📊", "title": "Upload Data", "desc": "Import historical consumption data", "page": "pages/data_loader.py"},
            {"icon": "📋", "title": "Energy Survey", "desc": "Complete smart energy assessment", "page": "pages/survey.py"},
            {"icon": "🤖", "title": "AI Forecast", "desc": "Generate 12-month predictions", "page": "pages/forecast.py"},
            {"icon": "💡", "title": "Optimization", "desc": "Get personalized recommendations", "page": "pages/optimization.py"},
            {"icon": "☀️", "title": "Solar Analysis", "desc": "Calculate solar ROI and savings", "page": "pages/solar.py"}
        ]
        
        for action in quick_actions:
            action_html = f"""
            <div class="quick-action-card">
                <div class="quick-action-icon">{action['icon']}</div>
                <div class="quick-action-title">{action['title']}</div>
                <div class="quick-action-desc">{action['desc']}</div>
            </div>
            """
            st.markdown(action_html, unsafe_allow_html=True)
            
            # Action button below each card
            if st.button(f"Go to {action['title']}", key=f"quick_{action['title']}", use_container_width=True):
                st.switch_page(action["page"])
        
        st.markdown("</div>", unsafe_allow_html=True)
       
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

if __name__ == "__main__":
    main()

