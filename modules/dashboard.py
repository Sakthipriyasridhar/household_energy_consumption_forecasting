import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import config

def show():
    # Hero Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='padding: 20px; border-radius: 15px; background: linear-gradient(135deg, #2E86AB, #A23B72); color: white;'>
            <h1 style='margin: 0;'>⚡ Energy Optimizer AI</h1>
            <p style='font-size: 1.2rem; opacity: 0.9;'>
                Transform your energy consumption with AI-powered insights
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Avg. User Savings", "32%", "↑ 5% this month")
    
    st.divider()
    
    # Quick Start Cards
    st.subheader("🚀 Quick Start")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container():
            st.markdown('<div class="energy-card">', unsafe_allow_html=True)
            if st.button("📝 Start Energy Survey", use_container_width=True):
                st.switch_page("pages/survey.py")
            st.caption("Complete in 5 minutes")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="energy-card">', unsafe_allow_html=True)
            if st.button("📊 Get AI Forecast", use_container_width=True):
                st.switch_page("pages/forecasting.py")
            st.caption("85%+ accuracy")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="energy-card">', unsafe_allow_html=True)
            if st.button("💡 Optimization Tips", use_container_width=True):
                st.switch_page("pages/optimization.py")
            st.caption("Save 15-40%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        with st.container():
            st.markdown('<div class="energy-card">', unsafe_allow_html=True)
            if st.button("☀️ Solar Analysis", use_container_width=True):
                st.switch_page("pages/solar.py")
            st.caption("ROI calculator")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # User Progress
    st.subheader("📈 Your Progress")
    
    if st.session_state.survey_completed:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_cost = st.session_state.user_data.get("monthly_cost", 0)
            st.metric("Current Monthly Cost", f"₹{current_cost:,.0f}")
        
        with col2:
            potential_savings = current_cost * 0.35  # Assume 35% savings
            st.metric("Potential Savings", f"₹{potential_savings:,.0f}/month")
        
        with col3:
            co2_reduction = (potential_savings / 8) * 0.85  # Approx CO2 kg
            st.metric("CO₂ Reduction", f"{co2_reduction:.1f} kg/month")
        
        # Consumption Breakdown
        st.subheader("🔋 Your Energy Breakdown")
        
        if "appliance_data" in st.session_state.user_data:
            df_appliances = pd.DataFrame(st.session_state.user_data["appliance_data"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                fig = px.pie(
                    df_appliances, 
                    values='consumption_kwh', 
                    names='appliance',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                fig.update_layout(title="Appliance Consumption Breakdown")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Bar chart
                fig = px.bar(
                    df_appliances.nlargest(5, 'consumption_kwh'),
                    x='appliance',
                    y='consumption_kwh',
                    color='consumption_kwh',
                    color_continuous_scale='Viridis',
                    title="Top 5 Energy Consumers"
                )
                fig.update_layout(yaxis_title="kWh/month")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Start with the Energy Survey to see personalized insights!")
    
    # AI Insights Section
    st.divider()
    st.subheader("🤖 AI Insights")
    
    insights = [
        {
            "title": "Peak Hour Optimization",
            "description": "Shift 30% of your consumption to off-peak hours",
            "savings": "Save ₹1,200/month",
            "icon": "🕒"
        },
        {
            "title": "Appliance Upgrade",
            "description": "Replace old refrigerator with 5-star rated model",
            "savings": "Save ₹800/month",
            "icon": "🔄"
        },
        {
            "title": "Solar Potential",
            "description": "Your roof can generate 450 kWh/month",
            "savings": "75% bill reduction",
            "icon": "☀️"
        }
    ]
    
    for insight in insights:
        with st.container():
            st.markdown(f"""
            <div class="energy-card">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div style="font-size: 2rem;">{insight['icon']}</div>
                    <div style="flex: 1;">
                        <h4 style="margin: 0; color: {config.THEME['primary']};">{insight['title']}</h4>
                        <p style="margin: 5px 0; color: {config.THEME['text_light']};">{insight['description']}</p>
                        <p style="margin: 0; font-weight: bold; color: {config.THEME['success']};">{insight['savings']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.caption(f"© {datetime.now().year} Energy Optimizer AI | v{config.APP_CONFIG['version']}")
