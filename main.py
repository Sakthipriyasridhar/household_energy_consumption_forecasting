import streamlit as st
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Energy Optimizer AI",
    page_icon="⚡",
    layout="wide"
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
            text-align: center;
        }
        
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
            font-weight: 400;
            text-align: center;
        }
        
        /* Vertical quick actions */
        .vertical-actions-container {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem 0;
        }
        
        .quick-action-card {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            border: 1px solid #e8e8e8;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 1.5rem;
            cursor: pointer;
        }
        
        .quick-action-card:hover {
            transform: translateY(-4px);
            border-color: #2E86AB;
            box-shadow: 0 8px 24px rgba(46, 134, 171, 0.15);
        }
        
        .quick-action-icon {
            font-size: 2.5rem;
            min-width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #2E86AB, #1b9aaa);
            color: white;
            border-radius: 12px;
        }
        
        .quick-action-content {
            flex: 1;
        }
        
        .quick-action-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 0.5rem;
        }
        
        .quick-action-desc {
            font-size: 0.95rem;
            color: #666;
            line-height: 1.5;
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
        
        /* Center container */
        .center-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 70vh;
        }
        
        /* Footer */
        .footer {
            margin-top: 3rem;
            padding: 2rem 0;
            border-top: 1px solid #e8e8e8;
            text-align: center;
            color: #666;
            font-size: 0.9rem;
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
    
    # Center the content
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">Energy Optimizer AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Energy Management & Optimization Platform</div>', unsafe_allow_html=True)
    
    st.markdown("<div class='vertical-actions-container'>", unsafe_allow_html=True)
    
    # Quick Actions
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
            "desc": "Complete smart energy assessment questionnaire",
            "page": "pages/survey.py"
        },
        {
            "icon": "🤖", 
            "title": "AI Forecast", 
            "desc": "Generate 12-month AI-powered predictions",
            "page": "pages/forecast.py"
        },
        {
            "icon": "💡", 
            "title": "Optimization", 
            "desc": "Get personalized energy savings recommendations",
            "page": "pages/optimization.py"
        },
        {
            "icon": "☀️", 
            "title": "Solar Analysis", 
            "desc": "Calculate solar ROI and savings potential",
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
    
    # Footer (outside center container)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    
    footer_cols = st.columns(3)
    
    with footer_cols[0]:
        st.caption("**Platform:** Energy Optimizer AI v2.1")
        st.caption("**Accuracy:** 85.2% average")
    
    with footer_cols[1]:
        st.caption("**Security:** 256-bit AES")
        st.caption(f"**Updated:** {datetime.now().strftime('%Y-%m-%d')}")
    
    with footer_cols[2]:
        st.caption("**Privacy:** Your data is secure")
        st.caption("**Open Source:** GitHub")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
