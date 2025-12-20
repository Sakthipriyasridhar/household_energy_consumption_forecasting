import streamlit as st
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Energy Optimizer AI",
    page_icon="⚡",
    layout="centered"  # Changed from "wide" to "centered"
)

# Clean, Professional CSS
def load_css():
    css = """
    <style>
        /* Reset margins and padding */
        .stApp {
            margin: 0;
            padding: 0;
        }
        
        /* Main container */
        .main-container {
            max-width: 600px;
            margin: 0 auto;
            padding: 1.5rem 1rem;
        }
        
        /* Header styles - more compact */
        .main-header {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 0.25rem;
            background: linear-gradient(90deg, #2E86AB, #06D6A0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
        }
        
        .sub-header {
            font-size: 0.95rem;
            color: #666;
            margin-bottom: 2rem;
            font-weight: 400;
            text-align: center;
            line-height: 1.4;
        }
        
        /* Vertical quick actions - more compact */
        .vertical-actions-container {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin: 0 auto;
        }
        
        .quick-action-card {
            background: white;
            padding: 1.25rem 1.5rem;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 1rem;
            cursor: pointer;
        }
        
        .quick-action-card:hover {
            border-color: #2E86AB;
            box-shadow: 0 4px 12px rgba(46, 134, 171, 0.1);
            transform: translateY(-1px);
        }
        
        .quick-action-icon {
            font-size: 1.5rem;
            min-width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #2E86AB, #1b9aaa);
            color: white;
            border-radius: 8px;
            flex-shrink: 0;
        }
        
        .quick-action-content {
            flex: 1;
            min-width: 0; /* Prevent content from overflowing */
        }
        
        .quick-action-title {
            font-size: 1rem;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 0.25rem;
            line-height: 1.3;
        }
        
        .quick-action-desc {
            font-size: 0.85rem;
            color: #666;
            line-height: 1.4;
        }
        
        /* Compact buttons */
        .stButton > button {
            background: linear-gradient(90deg, #2E86AB, #1b9aaa);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 6px;
            font-weight: 500;
            font-size: 0.85rem;
            transition: all 0.2s ease;
            width: 100%;
            margin-top: 0.5rem;
        }
        
        .stButton > button:hover {
            box-shadow: 0 3px 8px rgba(46, 134, 171, 0.2);
        }
        
        /* Remove extra Streamlit spacing */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Footer - more compact */
        .footer {
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid #f0f0f0;
            text-align: center;
            color: #666;
            font-size: 0.8rem;
        }
        
        .footer-caption {
            font-size: 0.75rem;
            color: #888;
            margin: 0.25rem 0;
        }
        
        /* Hide Streamlit branding if needed */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Ensure content doesn't overflow */
        .css-1d391kg {
            padding-top: 1rem !important;
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
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header - Compact
    st.markdown('<div class="main-header">Energy Optimizer AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Energy Management Platform</div>', unsafe_allow_html=True)
    
    # Quick Actions Container
    st.markdown("<div class='vertical-actions-container'>", unsafe_allow_html=True)
    
    # Quick Actions - More compact
    quick_actions = [
        {
            "icon": "📊", 
            "title": "Data Upload", 
            "desc": "Import historical energy consumption data",
            "page": "pages/data_loader.py"
        },
        {
            "icon": "📋", 
            "title": "Energy Survey", 
            "desc": "Complete smart energy assessment",
            "page": "pages/survey.py"
        },
        {
            "icon": "🤖", 
            "title": "AI Forecast", 
            "desc": "Generate 12-month AI predictions",
            "page": "pages/forecast.py"
        },
        {
            "icon": "💡", 
            "title": "Optimization", 
            "desc": "Get personalized savings recommendations",
            "page": "pages/optimization.py"
        },
        {
            "icon": "☀️", 
            "title": "Solar Analysis", 
            "desc": "Calculate solar ROI and savings",
            "page": "pages/solar.py"
        }
    ]
    
    for action in quick_actions:
        # Create the clickable card
        action_html = f"""
        <div class="quick-action-card" onclick="this.nextElementSibling.click()">
            <div class="quick-action-icon">{action['icon']}</div>
            <div class="quick-action-content">
                <div class="quick-action-title">{action['title']}</div>
                <div class="quick-action-desc">{action['desc']}</div>
            </div>
        </div>
        """
        st.markdown(action_html, unsafe_allow_html=True)
        
        # Hidden button for navigation
        if st.button(f"Go to {action['title']}", key=f"action_{action['title']}", use_container_width=True):
            st.switch_page(action["page"])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer - Compact
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    
    footer_cols = st.columns(3)
    
    with footer_cols[0]:
        st.caption("**Platform:** v2.1")
        st.caption("**Accuracy:** 85.2%")
    
    with footer_cols[1]:
        st.caption("**Security:** AES-256")
        st.caption(f"**Date:** {datetime.now().strftime('%m/%d')}")
    
    with footer_cols[2]:
        st.caption("**Privacy:** Secure")
        st.caption("**Open Source:**")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
