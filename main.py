import streamlit as st

# Page Configuration - HIDE default elements
st.set_page_config(
    page_title="Energy Optimizer AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS to hide Streamlit branding and default elements
def load_css():
    css = """
    <style>
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Fix sidebar to always be visible */
        section[data-testid="stSidebar"] {
            position: fixed !important;
            height: 100vh !important;
        }
        
        /* Adjust main content area */
        .main .block-container {
            padding-left: 5rem;
        }
        
        /* Main title styling */
        .main-title {
            background: linear-gradient(45deg, #2E86AB, #A23B72);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 700;
            margin-top: -1rem;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #1a1a2e, #16213e);
            color: white;
            padding-top: 2rem;
        }
        
        /* Sidebar title */
        .sidebar-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 2rem;
            text-align: center;
            background: linear-gradient(45deg, #00b4d8, #0077b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Radio button styling */
        div[data-testid="stRadio"] > div {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
        }
        
        div[data-testid="stRadio"] label {
            color: white !important;
            font-weight: 500;
        }
        
        /* Card styling */
        .energy-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 1.5rem;
            border-radius: 15px;
            border-left: 4px solid #00b4d8;
            margin-bottom: 1rem;
            transition: all 0.3s;
            backdrop-filter: blur(10px);
        }
        
        .energy-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.15);
            box-shadow: 0 8px 20px rgba(0, 180, 216, 0.2);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(45deg, #00b4d8, #0077b6);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0, 180, 216, 0.4);
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
        }
        
        /* Progress indicators */
        .progress-indicator {
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
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
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Dashboard"

# Navigation Sidebar Component
def create_sidebar():
    """Create permanent sidebar navigation that appears on all pages"""
    with st.sidebar:
        # App Logo/Title
        st.markdown('<h1 class="sidebar-title">⚡ Energy AI</h1>', unsafe_allow_html=True)
        
        # Navigation Options - using radio for single selection
        st.markdown("### 📍 Navigate")
        page = st.radio(
            "",
            ["🏠 Dashboard", "📋 Energy Survey", "📊 AI Forecast", "⚡ Optimization", "☀️ Solar Analysis"],
            index=["🏠 Dashboard", "📋 Energy Survey", "📊 AI Forecast", "⚡ Optimization", "☀️ Solar Analysis"].index(
                st.session_state.current_page
            )
        )
        
        # Update current page in session state
        st.session_state.current_page = page
        
        st.divider()
        
        # User Progress Section
        st.markdown("### 📈 Your Progress")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.survey_completed:
                st.success("✅ Survey")
            else:
                st.warning("📝 Survey")
        
        with col2:
            if st.session_state.forecast_generated:
                st.success("✅ Forecast")
            else:
                st.info("📊 Forecast")
        
        # Quick Stats
        st.divider()
        st.markdown("### 💰 Current Stats")
        
        if st.session_state.user_data.get("monthly_cost"):
            st.metric(
                "Monthly Bill",
                f"₹{st.session_state.user_data.get('monthly_cost', 0):,.0f}"
            )
        else:
            st.info("Complete survey to see your stats")
        
        # Features Preview
        st.divider()
        with st.expander("✨ Features Overview", expanded=False):
            st.write("""
            - 🤖 **AI Forecasting** - 85%+ accuracy
            - 📋 **Smart Survey** - 5-min assessment
            - 💡 **Optimization** - Personalized tips
            - ☀️ **Solar Analysis** - ROI calculator
            - 📊 **Analytics** - Interactive charts
            """)
        
        # Footer in sidebar
        st.divider()
        st.caption("🔋 Powered by Advanced ML")
        st.caption("v1.0 | Secure & Private")

# Main App Logic
def main():
    load_css()
    init_session_state()
    
    # Create permanent sidebar
    create_sidebar()
    
    # Get current page from session state
    current_page = st.session_state.current_page
    
    # Map page names to actual page files
    page_mapping = {
        "🏠 Dashboard": "Dashboard",
        "📋 Energy Survey": "pages/survey.py",
        "📊 AI Forecast": "pages/forecast.py", 
        "⚡ Optimization": "pages/optimization.py",
        "☀️ Solar Analysis": "pages/solar.py"
    }
    
    # Handle navigation - if not on Dashboard, switch page
    if current_page != "🏠 Dashboard":
        target_page = page_mapping[current_page]
        # Add a small delay for better UX
        with st.spinner(f"Loading {current_page[2:]}..."):
            st.switch_page(target_page)
    else:
        # Show Dashboard content
        show_dashboard()

def show_dashboard():
    """Dashboard Page - Home content"""
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h1 class="main-title">⚡ Energy Optimizer AI</h1>', unsafe_allow_html=True)
        st.markdown("### ML-Powered Household Energy Management System")
    
    with col2:
        st.write("")  # Spacer
    
    st.divider()
    
    # Hero Section with metrics
    st.subheader("📊 Dashboard Overview")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown("""
        <div class="metric-card">
            <h3>32%</h3>
            <p>Avg. Savings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown("""
        <div class="metric-card">
            <h3>85.2%</h3>
            <p>Forecast Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
        
    with metric_col3:
        st.markdown("""
        <div class="metric-card">
            <h3>12.5T</h3>
            <p>CO₂ Reduced</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        st.markdown("""
        <div class="metric-card">
            <h3>₹2,500</h3>
            <p>Avg. Monthly Save</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Quick Actions Section
    st.subheader("🚀 Quick Actions")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("📝 Start Energy Survey", use_container_width=True, key="survey_action"):
            st.session_state.current_page = "📋 Energy Survey"
            st.rerun()
    
    with action_col2:
        if st.button("📊 Generate Forecast", use_container_width=True, key="forecast_action"):
            st.session_state.current_page = "📊 AI Forecast"
            st.rerun()
    
    with action_col3:
        if st.button("💡 Get Tips", use_container_width=True, key="tips_action"):
            st.session_state.current_page = "⚡ Optimization"
            st.rerun()
    
    st.divider()
    
    # Features Grid
    st.subheader("✨ Key Features")
    
    features = [
        {"icon": "🤖", "title": "AI Forecasting", "desc": "12-month predictions with 85%+ accuracy", "page": "📊 AI Forecast"},
        {"icon": "📋", "title": "Smart Survey", "desc": "5-min comprehensive energy assessment", "page": "📋 Energy Survey"},
        {"icon": "💡", "title": "Personalized Tips", "desc": "Actionable savings recommendations", "page": "⚡ Optimization"},
        {"icon": "☀️", "title": "Solar Analysis", "desc": "ROI and payback period calculations", "page": "☀️ Solar Analysis"},
        {"icon": "📊", "title": "Real Analytics", "desc": "Interactive charts and insights", "page": "📊 AI Forecast"},
        {"icon": "💰", "title": "Cost Savings", "desc": "15-40% reduction potential", "page": "⚡ Optimization"}
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
                    <h4 style="margin-top: 0; color: white;">{feature['title']}</h4>
                    <p style="color: #b0b0b0; font-size: 0.9rem;">{feature['desc']}</p>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Add a button to go to the feature
                if st.button(f"Go to {feature['title']}", key=f"feature_btn_{idx}", use_container_width=True):
                    st.session_state.current_page = feature['page']
                    st.rerun()
    
    st.divider()
    
    # Recent Activity Section
    st.subheader("📈 Recent Insights")
    
    insight_tab1, insight_tab2, insight_tab3 = st.tabs(["💡 Tips", "📊 Usage", "🌱 Impact"])
    
    with insight_tab1:
        st.write("""
        **This Week's Top Tips:**
        
        1. **Switch to LED bulbs** - Save up to 75% on lighting
        2. **Use smart power strips** - Eliminate phantom loads
        3. **Optimize AC temperature** - Set to 24°C for 20% savings
        4. **Run appliances off-peak** - Save 15-20% on time-based rates
        """)
        
        if st.button("View All Tips", key="view_tips"):
            st.session_state.current_page = "⚡ Optimization"
            st.rerun()
    
    with insight_tab2:
        st.write("""
        **Usage Patterns:**
        
        - ⚡ **Peak Hours:** 6-10 PM (40% higher rates)
        - 🏠 **Avg. Monthly Use:** 900-1200 kWh
        - 🔥 **Top Consumers:** AC (40%), Water Heater (15%), Refrigerator (10%)
        - 📱 **Standby Power:** 5-10% of total bill
        """)
        
        if st.button("Analyze My Usage", key="analyze_usage"):
            st.session_state.current_page = "📊 AI Forecast"
            st.rerun()
    
    with insight_tab3:
        st.write("""
        **Environmental Impact:**
        
        - 🌍 **CO₂ Reduced:** Equivalent to 15 trees planted
        - 💧 **Water Saved:** 2,500 liters monthly
        - 🏭 **Clean Energy:** Potential for 40% solar coverage
        - 📉 **Waste Reduced:** 15% less energy waste
        """)
        
        if st.button("Calculate My Impact", key="calculate_impact"):
            st.session_state.current_page = "☀️ Solar Analysis"
            st.rerun()
    
    # Footer at bottom
    st.divider()
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
    with footer_col2:
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            <p>© 2024 Energy Optimizer AI | Advanced Machine Learning Platform</p>
            <p>🔒 Your data is encrypted and never shared with third parties</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
