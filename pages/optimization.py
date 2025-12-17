import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Optimization", page_icon="💡", layout="wide")
st.title("💡 Smart Energy Optimization")

# SIMPLIFIED CSS - FIXED
st.markdown("""
<style>
    /* SIMPLER CSS - No complex gradients */
    .priority-high {
        background-color: #ff6b6b;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
        display: inline-block;
    }
    .priority-medium {
        background-color: #ffd166;
        color: #333;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
        display: inline-block;
    }
    .priority-low {
        background-color: #06d6a0;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
        display: inline-block;
    }
    .savings-badge {
        background-color: #73AB84;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1em;
        display: inline-block;
    }
    /* Use Streamlit's built-in containers instead */
    .stContainer {
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Check if survey completed
if not st.session_state.get("survey_completed", False):
    st.warning("Optimization recommendations require your appliance usage data from the survey.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("📝 Take Survey Now", type="primary", use_container_width=True):
            st.switch_page("pages/survey.py")
    
    with col2:
        st.info("The survey takes only 5 minutes and provides personalized recommendations.")
    
    st.stop()

# ==================== USER HAS COMPLETED SURVEY ====================
st.success("✅ Showing personalized recommendations based on your survey data")

# Get user data
user_data = st.session_state.get("user_data", {})
monthly_cost = user_data.get("monthly_cost", 0)
monthly_consumption = user_data.get("monthly_consumption", 0)
appliances = user_data.get("appliances", [])
location = user_data.get("household", {}).get("location", "Unknown")

# Header with user summary
st.subheader(f"👋 Welcome back, {user_data.get('name', 'User')}!")
st.caption(f"📍 Location: {location} | 📊 Monthly Consumption: {monthly_consumption} kWh | 💰 Monthly Cost: ₹{monthly_cost:,.0f}")

# Calculate potential savings
potential_savings = monthly_cost * 0.35  # Assume 35% savings potential
optimized_cost = monthly_cost - potential_savings

# Display Summary Dashboard - USING STREAMLIT'S BUILT-IN METRICS
st.subheader("📊 Your Energy Savings Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Current Monthly Cost", f"₹{monthly_cost:,.0f}")

with col2:
    st.metric("Potential Monthly Savings", f"₹{potential_savings:,.0f}")

with col3:
    savings_percent = (potential_savings / monthly_cost * 100) if monthly_cost > 0 else 0
    st.metric("Optimized Cost", f"₹{optimized_cost:,.0f}")

with col4:
    annual_savings = potential_savings * 12
    st.metric("Annual Savings Potential", f"₹{annual_savings:,.0f}")

# Personalized Recommendations - USING STREAMLIT'S BUILT-IN COMPONENTS
st.subheader("🎯 Smart Recommendations Engine")

# Generate recommendations based on user data
recommendations = []

# Check for AC in appliances
for appliance in appliances:
    name = appliance.get("name", "").lower()
    quantity = appliance.get("quantity", 0)
    hours = appliance.get("hours", 0)
    
    if ("air" in name or "ac" in name) and quantity > 0:
        recommendations.append({
            "title": "Air Conditioner Optimization",
            "description": f"You have {quantity} AC unit(s) running {hours} hours/day",
            "action": "Increase temperature by 2°C, use sleep mode, clean filters monthly",
            "savings": f"₹{min(800, hours * 50)}/month",
            "priority": "High",
            "icon": "❄️",
            "category": "Cooling"
        })
    
    if "refrigerator" in name:
        recommendations.append({
            "title": "Refrigerator Efficiency",
            "description": "Consider upgrading to energy-efficient model",
            "action": "Set temperature to 4-5°C, ensure proper ventilation",
            "savings": "₹300-500/month",
            "priority": "Medium",
            "icon": "🧊",
            "category": "Kitchen"
        })

# Add general recommendations
recommendations.extend([
    {
        "title": "Lighting Upgrade",
        "description": "Replace incandescent bulbs with LED",
        "action": "Switch all bulbs to LED (10W vs 60W savings)",
        "savings": "₹200-300/month",
        "priority": "Low",
        "icon": "💡",
        "category": "Lighting"
    },
    {
        "title": "Solar Water Heater",
        "description": f"Excellent solar potential in {location}",
        "action": "Install 100L solar water heater",
        "savings": "₹300-500/month",
        "priority": "Medium",
        "icon": "☀️",
        "category": "Renewable"
    }
])

# Display recommendations using Streamlit components (NO COMPLEX HTML)
for rec in recommendations:
    with st.container():
        # Create columns for layout
        col_icon, col_content, col_savings = st.columns([1, 4, 1])
        
        with col_icon:
            st.markdown(f"<div style='font-size: 2.5em; text-align: center;'>{rec['icon']}</div>", unsafe_allow_html=True)
        
        with col_content:
            # Title with priority badge
            priority_color = {
                "High": "#ff6b6b",
                "Medium": "#ffd166", 
                "Low": "#06d6a0"
            }
            
            st.markdown(f"""
            **{rec['title']}** 
            <span style='background-color: {priority_color[rec['priority']]}; 
                         color: {'white' if rec['priority'] == 'High' else '#333'};
                         padding: 2px 10px;
                         border-radius: 10px;
                         font-size: 0.8em;
                         margin-left: 10px;'>
                {rec['priority']} Priority
            </span>
            """, unsafe_allow_html=True)
            
            st.caption(f"{rec['description']}")
            
            with st.expander("📋 Action Plan", expanded=False):
                st.info(f"**Action Required:** {rec['action']}")
                st.caption(f"Category: {rec['category']}")
        
        with col_savings:
            st.markdown(f"""
            <div style='
                background-color: #73AB84;
                color: white;
                padding: 10px;
                border-radius: 10px;
                text-align: center;
                margin: 10px 0;
            '>
                <strong>{rec['savings']}</strong><br>
                <small>Monthly Savings</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()

# Filter and Sort Options - SIMPLIFIED
st.subheader("🔍 Filter Recommendations")

col_filter1, col_filter2 = st.columns(2)
with col_filter1:
    priority_filter = st.multiselect(
        "Filter by Priority:",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )

# Filter recommendations
filtered_recommendations = [r for r in recommendations if r["priority"] in priority_filter]

if not filtered_recommendations:
    st.info("No recommendations match your filters. Try adjusting filter settings.")

# Implementation Timeline - USING STREAMLIT'S BUILT-IN COMPONENTS
st.subheader("📅 Implementation Timeline")

timeline_data = {
    "Immediate (Week 1-2)": [
        "Set AC temperature to 24°C",
        "Replace 5 highest-use bulbs with LED"
    ],
    "Short-term (Month 1)": [
        "Complete LED lighting conversion",
        "Install smart plugs"
    ],
    "Medium-term (Month 2-3)": [
        "Refrigerator energy audit",
        "Solar water heater assessment"
    ],
    "Long-term (Month 4-6)": [
        "Major appliance upgrades",
        "Solar PV system evaluation"
    ]
}

for phase, tasks in timeline_data.items():
    with st.expander(f"**{phase}**", expanded=True if phase == "Immediate (Week 1-2)" else False):
        for task in tasks:
            st.checkbox(task, value=False)

# Savings Calculator
st.subheader("💰 Savings Calculator")

if monthly_cost > 0:
    # Simple calculator
    col_calc1, col_calc2 = st.columns(2)
    
    with col_calc1:
        selected_count = st.slider(
            "Number of recommendations to implement:",
            1, len(recommendations), 3
        )
        
        investment = st.number_input(
            "Initial Investment (₹):",
            0, 100000, 10000, step=1000
        )
    
    with col_calc2:
        # Calculate estimated savings
        avg_savings = 350  # Average per recommendation
        monthly_savings = selected_count * avg_savings
        annual_savings = monthly_savings * 12
        roi_months = (investment / monthly_savings) if monthly_savings > 0 else 0
        
        st.metric("Estimated Monthly Savings", f"₹{monthly_savings:,.0f}")
        st.metric("ROI Period", f"{roi_months:.1f} months")
        
        if investment > 0:
            st.progress(min(monthly_savings / 1000, 1.0))
            st.caption(f"Monthly savings cover {monthly_savings/investment*100:.1f}% of investment value")

# Action Plan Generator
st.divider()
st.subheader("📋 Generate Action Plan")

if st.button("📥 Download Your Action Plan", type="primary", use_container_width=True):
    st.success("✅ Action plan generated successfully!")
    
    # Show what would be included
    with st.expander("📄 Plan Contents", expanded=True):
        st.markdown("""
        **Your Personalized Energy Action Plan Includes:**
        
        1. **Priority Recommendations** - Based on your usage patterns
        2. **Implementation Timeline** - Step-by-step schedule
        3. **Cost-Benefit Analysis** - ROI calculations
        4. **Vendor Contacts** - Trusted local suppliers
        5. **Government Subsidy Guide** - How to claim benefits
        6. **Monthly Tracking Sheet** - Monitor your progress
        
        *Note: In production, this would generate a downloadable PDF.*
        """)

# Monthly Tracking - SIMPLIFIED
st.subheader("📱 Track Your Progress")

track_col1, track_col2 = st.columns(2)

with track_col1:
    current_month = datetime.now().strftime("%B")
    st.markdown(f"### {current_month} Progress")
    
    implemented = st.number_input(
        "Recommendations implemented this month:",
        0, len(recommendations), 0
    )
    
    actual_savings = st.number_input(
        "Actual savings this month (₹):",
        0, 10000, 0
    )

with track_col2:
    if monthly_cost > 0:
        progress = (implemented / len(recommendations)) * 100
        st.metric("Implementation Progress", f"{progress:.0f}%")
        
        if actual_savings > 0:
            st.metric("Actual vs Target Savings", f"₹{actual_savings:,.0f}")
            
            # Environmental impact
            co2_saved = actual_savings / 8 * 0.85  # Estimate CO2 savings
            st.caption(f"🌱 Environmental impact: {co2_saved:.0f} kg CO₂ saved")

# Footer with tips
st.divider()
st.markdown("""
### 💡 Quick Tips for Energy Savings

1. **Start with behavioral changes** - They cost nothing and save immediately
2. **Focus on high-usage appliances** - AC, refrigerator, water heater
3. **Use timers and schedules** - Automate for consistency
4. **Regular maintenance** - Clean filters, check seals
5. **Monitor your consumption** - Awareness leads to savings

*Remember: Small changes add up to big savings over time!*
""")

# Refresh button
if st.button("🔄 Refresh Recommendations", use_container_width=True):
    st.rerun()
