import streamlit as st
from streamlit_option_menu import option_menu
import config
from modules import dashboard, survey, forecasting, optimization, solar

# Page Configuration
st.set_page_config(
    page_title="Energy Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injection
def inject_css():
    with open("assets/style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State
def init_session_state():
    if "user_data" not in st.session_state:
        st.session_state.user_data = {}
    if "survey_completed" not in st.session_state:
        st.session_state.survey_completed = False
    if "forecast_data" not in st.session_state:
        st.session_state.forecast_data = None

# Main App
def main():
    # Inject custom CSS
    inject_css()
    
    # Initialize session state
    init_session_state()
    
    # Sidebar Navigation
    with st.sidebar:
        st.image("assets/images/logo.png", width=150)  # Add your logo
        
        selected = option_menu(
            menu_title="Energy Optimizer AI",
            options=["Dashboard", "Energy Survey", "AI Forecast", "Optimization", "Solar Analysis", "Reports"],
            icons=["house", "clipboard-data", "graph-up", "lightbulb", "sun", "file-earmark-text"],
            menu_icon="lightning-charge",
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": config.THEME["background"]},
                "icon": {"color": config.THEME["primary"], "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px"},
                "nav-link-selected": {"background-color": config.THEME["primary"]},
            }
        )
        
        # User Progress Indicator
        if st.session_state.survey_completed:
            st.success("✅ Survey Completed")
        if st.session_state.forecast_data:
            st.info("📊 Forecast Generated")
        
        # Quick Stats in Sidebar
        st.divider()
        if st.session_state.user_data.get("monthly_consumption"):
            st.metric(
                "Current Monthly Cost",
                f"₹{st.session_state.user_data.get('monthly_cost', 0):,.0f}",
                help="Based on your survey data"
            )
    
    # Page Routing
    if selected == "Dashboard":
        dashboard.show()
    elif selected == "Energy Survey":
        survey.show()
    elif selected == "AI Forecast":
        forecasting.show()
    elif selected == "Optimization":
        optimization.show()
    elif selected == "Solar Analysis":
        solar.show()
    elif selected == "Reports":
        st.title("📄 Reports & Export")
        # Add reports module

if __name__ == "__main__":
    main()
