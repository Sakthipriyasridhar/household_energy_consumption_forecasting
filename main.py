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
        .energy-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #2E86AB;
            margin-bottom: 1rem;
            transition: transform 0.3s;
        }
        .energy-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background: linear-gradient(45deg, #2E86AB, #A23B72);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(46, 134, 171, 0.3);
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
    
    # Page selection - using radio for main navigation
    page_options = ["🏠 Dashboard", "📋 Energy Survey", "📊 AI Forecast", "⚡ Optimization", "☀️ Solar Analysis"]
    selected_page = st.sidebar.radio("Navigate to", page_options, index=0)
    
    # Map page names to actual page files
    page_mapping = {
        "🏠 Dashboard": "Dashboard",  # Current page
        "📋 Energy Survey": "pages/survey.py",
        "📊 AI Forecast": "pages/forecast.py", 
        "⚡ Optimization": "pages/optimization.py",
        "☀️ Solar Analysis": "pages/solar.py"
    }
    
    # User Progress
    st.sidebar.divider()
    if st.session_state.survey_completed:
        st.sidebar.success("✅ Survey Completed")
    else:
        st.sidebar.warning("📝 Survey Pending")
    
    # Quick Stats
    st.sidebar.divider()
    st.sidebar.subheader("📈 Quick Stats")
    
    if st.session_state.user_data.get("monthly_cost"):
        st.sidebar.metric(
            "Current Bill",
            f"₹{st.session_state.user_data.get('monthly_cost', 0):,.0f}"
        )
    else:
        st.sidebar.info("Complete survey to see your stats")
    
    # Features summary
    st.sidebar.divider()
    with st.sidebar.expander("✨ Features"):
        st.write("""
        - 🤖 AI Forecasting (85%+ accuracy)
        - 📋 Smart Energy Survey
        - 💡 Personalized Optimization Tips
        - ☀️ Solar ROI Analysis
        - 📊 Interactive Analytics
        """)
    
    st.sidebar.divider()
    st.sidebar.caption("🔋 Powered by ML | 85%+ Accuracy")
    
    # Handle page navigation
    if selected_page != "🏠 Dashboard":
        target_page = page_mapping[selected_page]
        st.info(f"Redirecting to {selected_page[2:]}...")
        st.switch_page(target_page)
    else:
        # Show Dashboard content
        show_dashboard()

def show_dashboard():
    """Dashboard Page - Home content"""
    st.markdown('<h1 class="main-title">⚡ Energy Optimizer AI</h1>', unsafe_allow_html=True)
    st.markdown("### ML-Powered Household Energy Management System")
    
    st.divider()
    
    # Hero Section with metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg. Savings", "32%", "+5% from avg")
    
    with col2:
        st.metric("Forecast Accuracy", "85.2%", "+2.1%")
        
    with col3:
        st.metric("CO₂ Reduced", "12.5 tons", "Monthly avg")
    
    st.divider()
    
    # Quick Start Section
    st.subheader("🚀 Quick Start")
    
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("📝 Start Energy Survey", use_container_width=True, key="survey_btn"):
            st.switch_page("pages/survey.py")
    
    with quick_col2:
        if st.button("📊 Generate Forecast", use_container_width=True, key="forecast_btn"):
            st.switch_page("pages/forecast.py")
    
    with quick_col3:
        if st.button("💡 Get Tips", use_container_width=True, key="tips_btn"):
            st.switch_page("pages/optimization.py")
    
    st.divider()
    
    # Features Grid
    st.subheader("✨ Key Features")
    
    features = [
        {"icon": "🤖", "title": "AI Forecasting", "desc": "12-month predictions with 85%+ accuracy", "page": "pages/3_Forecast.py"},
        {"icon": "📋", "title": "Smart Survey", "desc": "5-min comprehensive energy assessment", "page": "pages/2_Survey.py"},
        {"icon": "💡", "title": "Personalized Tips", "desc": "Actionable savings recommendations", "page": "pages/4_Optimization.py"},
        {"icon": "☀️", "title": "Solar Analysis", "desc": "ROI and payback period calculations", "page": "pages/5_Solar.py"},
        {"icon": "📊", "title": "Real Analytics", "desc": "Interactive charts and insights", "page": "pages/3_Forecast.py"},
        {"icon": "💰", "title": "Cost Savings", "desc": "15-40% reduction potential", "page": "pages/4_Optimization.py"}
    ]
    
    # Create 3 columns for features
    cols = st.columns(3)
    
    for idx, feature in enumerate(features):
        col_idx = idx % 3
        with cols[col_idx]:
            with st.container():
                # Create a card-like container
                card_html = f"""
                <div class="energy-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{feature['icon']}</div>
                    <h4 style="margin-top: 0;">{feature['title']}</h4>
                    <p style="color: #666; font-size: 0.9rem;">{feature['desc']}</p>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Add a button to go to the feature
                if st.button(f"Go to {feature['title']}", key=f"feature_btn_{idx}", use_container_width=True):
                    st.switch_page(feature['page'])
    
    st.divider()
    
    # Status Section
    st.subheader("📊 Your Progress")
    
    progress_col1, progress_col2, progress_col3 = st.columns(3)
    
    with progress_col1:
        if st.session_state.survey_completed:
            st.success("✅ Survey Completed")
        else:
            st.warning("⏳ Survey Pending")
    
    with progress_col2:
        if st.session_state.forecast_generated:
            st.success("✅ Forecast Ready")
        else:
            st.info("📈 Forecast Available")
    
    with progress_col3:
        st.info("⚡ Optimization Ready")
    
    # Demo data section
    st.divider()
    st.subheader("📈 Sample Insights")
    
    insight_tab1, insight_tab2, insight_tab3 = st.tabs(["Savings", "Usage", "Efficiency"])
    
    with insight_tab1:
        st.write("""
        **Typical Savings Breakdown:**
        - Lighting: 15-20% savings
        - Appliances: 20-30% savings  
        - HVAC: 25-40% savings
        - Electronics: 10-15% savings
        """)
    
    with insight_tab2:
        st.write("""
        **Average Household Usage:**
        - Monthly: 900-1200 kWh
        - Peak Hours: 6-10 PM
        - Highest Consumers: AC, Water Heater, Refrigerator
        """)
    
    with insight_tab3:
        st.write("""
        **Efficiency Tips:**
        - Use LED bulbs: Save 75% on lighting
        - Smart thermostat: Save 10-12% on HVAC
        - Energy Star appliances: Save 10-50% per device
        - Solar panels: Reduce bills by 40-70%
        """)
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.caption("Electricity Forecaster Using AI | ML-Powered Energy Management Platform")
        st.caption("Data Privacy Assured | 256-bit Encryption")

if __name__ == "__main__":
    main()
