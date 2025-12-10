import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Energy Optimizer AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    css = """
    <style>
        .main-title {
            background: linear-gradient(45deg, #2E86AB, #A23B72);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 700;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #F5F7FA, #FFFFFF);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Initialize Session State
def init_session_state():
    if "survey_completed" not in st.session_state:
        st.session_state.survey_completed = False
    if "user_data" not in st.session_state:
        st.session_state.user_data = {}
    if "forecast_generated" not in st.session_state:
        st.session_state.forecast_generated = False

# Main App
def main():
    load_css()
    init_session_state()
    
    # Sidebar Navigation
    st.sidebar.title("⚡ Energy Optimizer AI")
    
    # Navigation Options
    page = st.sidebar.radio(
        "Navigate to",
        ["🏠 Dashboard", "📋 Energy Survey", "🤖 AI Forecast", "💡 Optimization", "☀️ Solar Analysis"],
        index=0
    )
    
    # User Progress
    st.sidebar.divider()
    if st.session_state.survey_completed:
        st.sidebar.success("✅ Survey Completed")
    else:
        st.sidebar.info("📝 Survey Pending")
    
    # Quick Stats
    if st.session_state.user_data.get("monthly_cost"):
        st.sidebar.metric(
            "Current Bill",
            f"₹{st.session_state.user_data['monthly_cost']:,.0f}"
        )
    
    # Display Current Page
    st.sidebar.divider()
    st.sidebar.caption("Powered by ML | 85%+ Accuracy")
    
    # Main Content based on selection
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📋 Energy Survey":
        # Will be handled by pages/2_Survey.py
        st.write("")  # Placeholder - actual content in pages
    elif page == "🤖 AI Forecast":
        # Will be handled by pages/3_Forecast.py
        st.write("")
    elif page == "💡 Optimization":
        # Will be handled by pages/4_Optimization.py
        st.write("")
    elif page == "☀️ Solar Analysis":
        # Will be handled by pages/5_Solar.py
        st.write("")

def show_dashboard():
    """Dashboard Page"""
    st.markdown('<h1 class="main-title">⚡ Energy Optimizer AI</h1>', unsafe_allow_html=True)
    st.markdown("### ML-Powered Household Energy Management")
    
    st.divider()
    
    # Hero Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg. Savings", "32%", "↑ 5%")
    
    with col2:
        st.metric("Forecast Accuracy", "85.2%", "↑ 2.1%")
        
    with col3:
        st.metric("CO₂ Reduced", "12.5 tons", "Monthly")
    
    st.divider()
    
    # Quick Start Cards
    st.subheader("🚀 Get Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Start Energy Survey", use_container_width=True):
            st.switch_page("pages/survey.py")
    
    with col2:
        if st.button("📊 View Sample Forecast", use_container_width=True):
            st.switch_page("pages/forecast.py")
    
    with col3:
        if st.button("💡 See Optimization Tips", use_container_width=True):
            st.switch_page("pages/optimization.py")
    
    st.divider()
    
    # Features
    st.subheader("✨ Key Features")
    
    features = [
        {"icon": "🤖", "title": "AI Forecasting", "desc": "12-month predictions with 85%+ accuracy"},
        {"icon": "📋", "title": "Smart Survey", "desc": "5-minute comprehensive energy assessment"},
        {"icon": "💡", "title": "Personalized Tips", "desc": "Actionable savings recommendations"},
        {"icon": "☀️", "title": "Solar Analysis", "desc": "ROI and payback period calculations"},
        {"icon": "📊", "title": "Real Analytics", "desc": "Interactive charts and insights"},
        {"icon": "💰", "title": "Cost Savings", "desc": "15-40% reduction potential"}
    ]
    
    cols = st.columns(3)
    for idx, feature in enumerate(features):
        with cols[idx % 3]:
            with st.container():
                st.markdown(f"""
                <div class="energy-card">
                    <div style="font-size: 2rem;">{feature['icon']}</div>
                    <h4>{feature['title']}</h4>
                    <p style="color: #666;">{feature['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.caption("© Energy Forecasting AI | ML-Powered Energy Management Platform")

if __name__ == "__main__":
    main()




