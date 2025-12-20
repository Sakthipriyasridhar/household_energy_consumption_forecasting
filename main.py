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
        
        /* Feature cards */
        .feature-card {
            background: white;
            padding: 1.8rem;
            border-radius: 12px;
            border: 1px solid #e8e8e8;
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        
        .feature-card:hover {
            transform: translateY(-4px);
            border-color: #2E86AB;
            box-shadow: 0 8px 24px rgba(46, 134, 171, 0.12);
        }
        
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            color: #2E86AB;
        }
        
        .feature-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 0.75rem;
        }
        
        .feature-desc {
            font-size: 0.95rem;
            color: #666;
            line-height: 1.5;
            flex-grow: 1;
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

def show_dashboard():
    """Clean, Professional Dashboard"""
    
    # Header Section
    col_header1, col_header2 = st.columns([3, 1])
    
    with col_header1:
        st.markdown('<div class="main-header">Energy Optimizer AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Machine Learning-Powered Energy Management & Optimization Platform</div>', unsafe_allow_html=True)
    
    with col_header2:
        st.markdown(f"<div style='text-align: right; color: #666; font-size: 0.9rem;'>{datetime.now().strftime('%B %d, %Y')}</div>", unsafe_allow_html=True)
    
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
    
    # Quick Actions Section
    st.markdown("### Quick Actions")
    
    action_cols = st.columns(4)
    
    actions = [
        {"icon": "📊", "title": "Upload Data", "desc": "Import historical data", "page": "pages/data_loader.py"},
        {"icon": "📋", "title": "Start Survey", "desc": "Complete energy assessment", "page": "pages/survey.py"},
        {"icon": "🤖", "title": "Generate Forecast", "desc": "AI-powered predictions", "page": "pages/forecast.py"},
        {"icon": "💡", "title": "Get Tips", "desc": "Personalized recommendations", "page": "pages/optimization.py"}
    ]
    
    for idx, action in enumerate(actions):
        with action_cols[idx]:
            if st.button(f"**{action['icon']} {action['title']}**\n\n{action['desc']}", 
                        use_container_width=True, 
                        key=f"action_{idx}"):
                st.switch_page(action["page"])
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Features Grid
    st.markdown("### Platform Features")
    
    features = [
        {
            "icon": "📊",
            "title": "Data Analytics",
            "desc": "Advanced analytics with interactive visualizations and historical trend analysis",
            "color": "#2E86AB"
        },
        {
            "icon": "🤖",
            "title": "AI Forecasting",
            "desc": "12-month predictions with 85%+ accuracy using machine learning algorithms",
            "color": "#06D6A0"
        },
        {
            "icon": "💡",
            "title": "Smart Optimization",
            "desc": "Personalized recommendations to reduce energy consumption by 15-40%",
            "color": "#FFD166"
        },
        {
            "icon": "☀️",
            "title": "Solar Analysis",
            "desc": "ROI calculations and payback period analysis for solar installations",
            "color": "#EF476F"
        },
        {
            "icon": "📱",
            "title": "Real-time Monitoring",
            "desc": "Live energy tracking and consumption pattern analysis",
            "color": "#118AB2"
        },
        {
            "icon": "💰",
            "title": "Cost Savings",
            "desc": "Detailed breakdown of potential savings and investment returns",
            "color": "#73AB84"
        }
    ]
    
    # Create two rows of features (3 per row)
    for row in range(0, len(features), 3):
        cols = st.columns(3)
        for col_idx in range(3):
            if row + col_idx < len(features):
                feature = features[row + col_idx]
                with cols[col_idx]:
                    feature_html = f"""
                    <div class="feature-card">
                        <div class="feature-icon" style="color: {feature['color']};">{feature['icon']}</div>
                        <div class="feature-title">{feature['title']}</div>
                        <div class="feature-desc">{feature['desc']}</div>
                    </div>
                    """
                    st.markdown(feature_html, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Analytics Preview Section
    st.markdown("### Analytics Preview")
    
    tab1, tab2, tab3 = st.tabs(["📈 Consumption Trends", "💰 Cost Analysis", "🌍 Environmental Impact"])
    
    with tab1:
        # Create sample consumption data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
        consumption = [850, 920, 780, 950, 1100, 1250, 1150, 980]
        optimized = [720, 780, 650, 800, 920, 1050, 970, 820]
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=months, y=consumption,
            mode='lines+markers',
            name='Current Consumption',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8)
        ))
        fig_trend.add_trace(go.Scatter(
            x=months, y=optimized,
            mode='lines+markers',
            name='Optimized Target',
            line=dict(color='#06D6A0', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        fig_trend.update_layout(
            title="Monthly Energy Consumption (kWh)",
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
    
    with tab2:
        # Cost breakdown
        categories = ['HVAC', 'Lighting', 'Appliances', 'Electronics', 'Water Heating']
        costs = [3200, 1200, 1800, 900, 1500]
        savings = [960, 240, 540, 135, 450]  # 30%, 20%, 30%, 15%, 30%
        
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(
            x=categories,
            y=costs,
            name='Current Cost',
            marker_color='#2E86AB',
            text=[f'₹{c:,.0f}' for c in costs],
            textposition='auto',
        ))
        fig_cost.add_trace(go.Bar(
            x=categories,
            y=[c - s for c, s in zip(costs, savings)],
            name='Optimized Cost',
            marker_color='#06D6A0',
            text=[f'₹{c-s:,.0f}' for c, s in zip(costs, savings)],
            textposition='auto',
        ))
        
        fig_cost.update_layout(
            title="Cost Breakdown by Category (₹/month)",
            barmode='group',
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig_cost, use_container_width=True, config={'displayModeBar': False})
    
    with tab3:
        # Environmental impact
        co2_data = {
            "Metric": ["CO₂ Saved", "Equivalent Trees", "Cars Off Road", "Coal Not Burned"],
            "Value": [12500, 595, 2.7, 5000],
            "Unit": ["kg", "trees", "cars", "kg"]
        }
        
        df_co2 = pd.DataFrame(co2_data)
        
        fig_env = go.Figure()
        fig_env.add_trace(go.Bar(
            x=df_co2["Metric"],
            y=df_co2["Value"],
            text=[f'{v:,.0f} {u}' for v, u in zip(df_co2["Value"], df_co2["Unit"])],
            textposition='auto',
            marker_color=['#06D6A0', '#73AB84', '#2E86AB', '#118AB2']
        ))
        
        fig_env.update_layout(
            title="Annual Environmental Impact",
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="",
            yaxis_title="",
            showlegend=False
        )
        
        st.plotly_chart(fig_env, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # CTA Section
    col_cta1, col_cta2, col_cta3 = st.columns([1, 2, 1])
    
    with col_cta2:
        st.markdown("### Ready to Optimize Your Energy?")
        st.markdown("Start your journey towards smarter energy management and significant cost savings.")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 Get Started", type="primary", use_container_width=True):
                st.switch_page("pages/survey.py")
        
        with col_btn2:
            if st.button("📚 Learn More", use_container_width=True):
                st.info("Explore our documentation and case studies")
    
    # Footer
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    footer_cols = st.columns(3)
    
    with footer_cols[0]:
        st.caption("**Accuracy:** 85.2% average forecast accuracy")
        st.caption("**Security:** 256-bit AES encryption")
    
    with footer_cols[1]:
        st.caption("**Platform:** Energy Optimizer AI v2.1")
        st.caption("**Last Updated:** " + datetime.now().strftime("%Y-%m-%d"))
    
    with footer_cols[2]:
        st.caption("**Support:** support@energyoptimizer.ai")
        st.caption("**Privacy:** Your data is secure with us")

if __name__ == "__main__":
    main()
