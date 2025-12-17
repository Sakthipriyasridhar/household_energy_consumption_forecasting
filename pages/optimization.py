import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Optimization", page_icon="💡", layout="wide")
st.title("💡 Smart Energy Optimization")

# SIMPLIFIED CSS - FIXED
st.markdown("""
<style>
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

# Display Summary Dashboard
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

# Personalized Recommendations - FIXED DUPLICATE ISSUE
st.subheader("🎯 Smart Recommendations Engine")

# Generate recommendations based on user data
recommendations = []

# Track which recommendations we've already added
recommendation_titles = set()

# Check for AC in appliances - FIXED: Only add once
ac_count = 0
ac_total_hours = 0
has_ac = False

for appliance in appliances:
    name = appliance.get("name", "").lower()
    quantity = appliance.get("quantity", 0)
    hours = appliance.get("hours", 0)
    
    # Count ACs
    if ("air" in name or "ac" in name or "air conditioner" in name) and quantity > 0:
        ac_count += quantity
        ac_total_hours += hours
        has_ac = True

# Add AC recommendation only once if user has AC
if has_ac and "Air Conditioner Optimization" not in recommendation_titles:
    avg_ac_hours = ac_total_hours / ac_count if ac_count > 0 else 0
    recommendations.append({
        "title": "Air Conditioner Optimization",
        "description": f"You have {ac_count} AC unit(s) running {avg_ac_hours:.1f} hours/day on average",
        "action": "Increase temperature by 2°C, use sleep mode, clean filters monthly, use curtains during day",
        "savings": f"₹{min(800, avg_ac_hours * 60)}/month",
        "priority": "High",
        "icon": "❄️",
        "category": "Cooling"
    })
    recommendation_titles.add("Air Conditioner Optimization")

# Check for other appliances
for appliance in appliances:
    name = appliance.get("name", "").lower()
    quantity = appliance.get("quantity", 0)
    age = appliance.get("age", 0)
    
    # Refrigerator - only add once
    if ("refrigerator" in name or "fridge" in name) and "Refrigerator Efficiency" not in recommendation_titles:
        recommendations.append({
            "title": "Refrigerator Efficiency",
            "description": f"{age}-year old refrigerator - consider upgrade" if age > 5 else "Refrigerator efficiency improvements",
            "action": "Set temperature to 4-5°C, ensure proper ventilation, check door seals",
            "savings": "₹300-500/month",
            "priority": "Medium",
            "icon": "🧊",
            "category": "Kitchen"
        })
        recommendation_titles.add("Refrigerator Efficiency")
    
    # Water Heater - only add once
    if ("water heater" in name or "geyser" in name) and "Water Heater Optimization" not in recommendation_titles:
        recommendations.append({
            "title": "Water Heater Optimization",
            "description": "Electric water heater consumes significant energy",
            "action": "Install timer, reduce temperature to 50°C, use during off-peak hours",
            "savings": "₹200-400/month",
            "priority": "Medium",
            "icon": "🔥",
            "category": "Heating"
        })
        recommendation_titles.add("Water Heater Optimization")

# Add general recommendations (only if not already added)
general_recommendations = [
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
        "action": "Install 100L solar water heater for free hot water",
        "savings": "₹300-500/month",
        "priority": "Medium",
        "icon": "☀️",
        "category": "Renewable"
    },
    {
        "title": "Standby Power Reduction",
        "description": "Phantom load from devices on standby",
        "action": "Use smart plugs, turn off at switch when not in use",
        "savings": "₹100-200/month",
        "priority": "Low",
        "icon": "🔌",
        "category": "Behavioral"
    },
    {
        "title": "Peak Hour Management",
        "description": "Shift heavy usage to off-peak hours",
        "action": "Run washing machine, dishwasher after 10 PM when rates are lower",
        "savings": "₹150-250/month",
        "priority": "Medium",
        "icon": "🕒",
        "category": "Behavioral"
    }
]

# Add only unique general recommendations
for rec in general_recommendations:
    if rec["title"] not in recommendation_titles:
        recommendations.append(rec)
        recommendation_titles.add(rec["title"])

# Display recommendations using Streamlit components
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

# Filter and Sort Options
st.subheader("🔍 Filter Recommendations")

if recommendations:
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
else:
    st.info("No recommendations available. Please check your appliance data in the survey.")

# Implementation Timeline
st.subheader("📅 Implementation Timeline")

timeline_data = {
    "Immediate (Week 1-2)": [
        "Set AC temperature to 24°C",
        "Replace 5 highest-use bulbs with LED",
        "Install power strips for standby devices"
    ],
    "Short-term (Month 1)": [
        "Complete LED lighting conversion",
        "Install smart plugs for heavy appliances",
        "Implement off-peak scheduling"
    ],
    "Medium-term (Month 2-3)": [
        "Refrigerator energy audit",
        "Solar water heater assessment",
        "Whole-house energy monitoring setup"
    ],
    "Long-term (Month 4-6)": [
        "Major appliance upgrades if needed",
        "Solar PV system evaluation",
        "Home automation integration"
    ]
}

for phase, tasks in timeline_data.items():
    with st.expander(f"**{phase}**", expanded=True if phase == "Immediate (Week 1-2)" else False):
        for task in tasks:
            st.checkbox(task, value=False)

# Savings Calculator
st.subheader("💰 Savings Calculator")

if monthly_cost > 0 and recommendations:
    # Simple calculator
    col_calc1, col_calc2 = st.columns(2)
    
    with col_calc1:
        selected_count = st.slider(
            "Number of recommendations to implement:",
            1, len(recommendations), min(3, len(recommendations))
        )
        
        investment = st.number_input(
            "Initial Investment (₹):",
            0, 100000, 10000, step=1000
        )
    
    with col_calc2:
        # Calculate estimated savings based on selected recommendations
        avg_savings_per_rec = 350  # Average per recommendation
        monthly_savings = selected_count * avg_savings_per_rec
        annual_savings = monthly_savings * 12
        roi_months = (investment / monthly_savings) if monthly_savings > 0 else 0
        
        st.metric("Estimated Monthly Savings", f"₹{monthly_savings:,.0f}")
        st.metric("ROI Period", f"{roi_months:.1f} months")
        
        if investment > 0:
            progress = min(monthly_savings / 1000, 1.0)
            st.progress(progress)
            st.caption(f"Monthly savings: {monthly_savings/investment*100:.1f}% of investment value")

# Action Plan Generator
st.divider()
st.subheader("📋 Generate Action Plan")

if st.button("📥 Download Your Action Plan", type="primary", use_container_width=True):
    st.success("✅ Action plan generated successfully!")
    
    # Show what would be included
    with st.expander("📄 Plan Contents", expanded=True):
        st.markdown(f"""
        **Your Personalized Energy Action Plan Includes:**
        
        1. **{len(recommendations)} Priority Recommendations** - Based on your usage patterns
        2. **Implementation Timeline** - Step-by-step schedule over 6 months
        3. **Cost-Benefit Analysis** - ROI calculations for each recommendation
        4. **Vendor Contacts** - Trusted local suppliers in {location}
        5. **Government Subsidy Guide** - How to claim energy efficiency benefits
        6. **Monthly Tracking Sheet** - Monitor your progress
        
        **Total Estimated Monthly Savings:** ₹{potential_savings:,.0f}
        **Annual Savings Potential:** ₹{annual_savings:,.0f}
        
        *Note: In production, this would generate a downloadable PDF.*
        """)

# Monthly Tracking
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
        progress = (implemented / len(recommendations)) * 100 if recommendations else 0
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
