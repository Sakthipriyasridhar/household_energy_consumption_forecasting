import streamlit as st

st.set_page_config(
    page_title="Household Energy Analysis",
    page_icon="🏠",
    layout="wide"
)

st.title(" Household Electricity Consumption Forecasting")
st.markdown("### Your Complete Energy Analysis Solution")

# Check if user has completed survey
survey_completed = 'user_data' in st.session_state

# Progress tracker
st.subheader(" Recommended Workflow")
col1, col2, col3 = st.columns(3)

with col1:
    st.info(" **1. AI Forecasting**")
    st.write("Predict future consumption")

with col2:
    if survey_completed:
        st.success(" **2. Energy Survey**")
        st.write("Survey completed!")
    else:
        st.info(" **2. Energy Survey**")
        st.write("Start here to calculate your current bill")

with col3:
    if survey_completed:
        st.success(" **3. Optimization**")
        st.write("Get personalized recommendations")
    else:
        st.warning(" **3. Optimization**")
        st.write("Complete survey first")



# Main navigation cards
st.markdown("---")
st.subheader("🚀 Get Started")

main_col1, main_col2, main_col3 = st.columns(3)

with main_col1:
    st.markdown("### 🔮 AI Forecasting")
    st.write("""
    - Future bill predictions
    - Machine learning models
    - Seasonal trend analysis
    - Upload historical data
    """)
    if st.button("Start Forecasting", key="forecast_btn", use_container_width=True):
        st.switch_page("pages/forecast.py")

with main_col2:
    st.markdown("### 📋 Energy Survey")
    st.write("""
    - Calculate current electricity bill
    - Input appliance usage patterns
    - TNEB slab rate calculation
    - Daily consumption breakdown
    """)
    if st.button("Start Survey", key="survey_btn", use_container_width=True):
        st.switch_page("pages/survey.py")

with main_col3:
    st.markdown("### 💰 Optimization")
    st.write("""
    - Personalized savings recommendations
    - Solar panel potential analysis
    - Appliance-specific tips
    - Bill reduction strategies
    """)
    if survey_completed:
        if st.button("View Recommendations", key="opt_btn", use_container_width=True):
            st.switch_page("pages/optimization.py")
    else:
        st.button("Complete Survey First", key="opt_disabled", disabled=True, use_container_width=True)



# Quick stats if survey completed
if survey_completed:
    st.markdown("---")
    st.subheader("📊 Your Current Energy Profile")
    user_data = st.session_state.user_data

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("Monthly Consumption", f"{user_data['total_units']:.0f} kWh")
    with stat_col2:
        st.metric("Current Bill", f"₹{user_data['current_bill']:.0f}")
    with stat_col3:
        st.metric("Season", user_data['season'])
    with stat_col4:
        efficiency = "High" if user_data['total_units'] < 200 else "Medium" if user_data['total_units'] < 350 else "Low"
        st.metric("Efficiency", efficiency)

# Footer
st.markdown("---")

st.markdown("💡 **Tip:** Complete the Energy Survey first to unlock personalized optimization recommendations")
