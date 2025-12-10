import streamlit as st

st.set_page_config(page_title="Optimization", page_icon="💡")
st.title("💡 Optimization Recommendations")

# Check if survey completed
if not st.session_state.get("survey_completed", False):
    st.warning("Optimization recommendations require your appliance usage data from the survey.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("📝 Take Survey Now", type="primary", use_container_width=True):
            st.switch_page("pages/survey.py")
    
    with col2:
        st.info("The survey takes only 5 minutes and provides personalized recommendations.")
    
    # Show sample recommendations
    st.divider()
    st.subheader("📋 Sample Recommendations (What you'll get)")
    
    sample_recommendations = [
        {
            "appliance": "Air Conditioner",
            "current": "₹800/month",
            "optimized": "₹500/month",
            "savings": "₹300/month",
            "action": "Set temperature to 24°C, use timer",
            "priority": "High"
        },
        {
            "appliance": "Refrigerator (10+ years)",
            "current": "₹600/month",
            "optimized": "₹300/month",
            "savings": "₹300/month",
            "action": "Replace with 5-star rated model",
            "priority": "Medium"
        },
        {
            "appliance": "Lighting",
            "current": "₹400/month",
            "optimized": "₹150/month",
            "savings": "₹250/month",
            "action": "Switch all bulbs to LED",
            "priority": "Low"
        }
    ]
    
    for rec in sample_recommendations:
        with st.container():
            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #ffffff, #f8f9fa);
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
                border-left: 5px solid #2E86AB;
                box-shadow: 0 3px 10px rgba(0,0,0,0.08);
            '>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <h4 style='margin: 0; color: #2E86AB;'>{rec['appliance']}</h4>
                        <p style='margin: 5px 0;'><strong>Action:</strong> {rec['action']}</p>
                    </div>
                    <div style='text-align: right;'>
                        <p style='margin: 0;'><strong>Savings:</strong> <span style='color: #73AB84;'>{rec['savings']}</span></p>
                        <p style='margin: 0; font-size: 0.9em; color: #666;'>Priority: {rec['priority']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.stop()

# User has completed survey - Show personalized recommendations
st.success("✅ Showing personalized recommendations based on your survey data")

# Get user data
user_data = st.session_state.get("user_data", {})
monthly_cost = user_data.get("monthly_cost", 0)

# Calculate potential savings
potential_savings = monthly_cost * 0.35  # Assume 35% savings potential
optimized_cost = monthly_cost - potential_savings

# Display Summary
st.subheader("📊 Your Savings Potential")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Current Monthly Cost", f"₹{monthly_cost:,.0f}")

with col2:
    st.metric("Potential Savings", f"₹{potential_savings:,.0f}/month")

with col3:
    savings_percent = (potential_savings / monthly_cost * 100) if monthly_cost > 0 else 0
    st.metric("Savings Percentage", f"{savings_percent:.0f}%")

# Personalized Recommendations
st.subheader("🎯 Your Personalized Recommendations")

# Generate recommendations based on user data
recommendations = []

# Check for AC in appliances
appliances = user_data.get("appliances", [])
for appliance in appliances:
    if "Air Conditioner" in appliance.get("name", "") and appliance.get("quantity", 0) > 0:
        recommendations.append({
            "title": "Air Conditioner Optimization",
            "description": f"Your {appliance['quantity']} AC unit(s) running {appliance.get('hours', 8)} hours/day",
            "action": "Increase temperature by 2°C, use sleep mode, clean filters monthly",
            "savings": "₹400-600/month",
            "priority": "🔥 High",
            "icon": "❄️"
        })
    
    if "Refrigerator" in appliance.get("name", ""):
        recommendations.append({
            "title": "Refrigerator Efficiency",
            "description": "Old refrigerators consume 2-3x more energy",
            "action": "Consider upgrading to 5-star rated model, maintain 4-5°C temperature",
            "savings": "₹300-500/month",
            "priority": "🟡 Medium",
            "icon": "🧊"
        })

# Add general recommendations
recommendations.extend([
    {
        "title": "Lighting Upgrade",
        "description": "Replace incandescent bulbs with LED",
        "action": "Switch all bulbs to LED (10W vs 60W savings)",
        "savings": "₹200-300/month",
        "priority": "🟢 Low",
        "icon": "💡"
    },
    {
        "title": "Peak Hour Management",
        "description": "Shift heavy usage to off-peak hours",
        "action": "Run washing machine, dishwasher after 10 PM",
        "savings": "₹150-250/month",
        "priority": "🟡 Medium",
        "icon": "🕒"
    },
    {
        "title": "Standby Power Reduction",
        "description": "Appliances on standby still consume power",
        "action": "Use smart plugs, turn off at switch when not in use",
        "savings": "₹100-200/month",
        "priority": "🟢 Low",
        "icon": "🔌"
    }
])

# Display recommendations
for rec in recommendations:
    with st.container():
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            border: 1px solid #e0e0e0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        '>
            <div style='display: flex; align-items: start; gap: 15px;'>
                <div style='font-size: 2rem;'>{rec['icon']}</div>
                <div style='flex: 1;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <h4 style='margin: 0; color: #2E86AB;'>{rec['title']} <span style='font-size: 0.8em; background: #f0f0f0; padding: 2px 8px; border-radius: 10px;'>{rec['priority']}</span></h4>
                        <strong style='color: #73AB84;'>{rec['savings']}</strong>
                    </div>
                    <p style='margin: 8px 0; color: #666;'>{rec['description']}</p>
                    <div style='background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                        <strong>📌 Action Required:</strong> {rec['action']}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Implementation Timeline
st.subheader("📅 Recommended Implementation Timeline")

timeline_data = {
    "Week 1-2": ["Set AC temperature to 24°C", "Replace 5 bulbs with LED"],
    "Month 1": ["Install smart plugs for standby devices", "Schedule laundry for weekends"],
    "Month 2-3": ["Consider refrigerator upgrade", "Install water heater timer"],
    "Month 4-6": ["Evaluate solar feasibility", "Complete all lighting upgrades"]
}

for timeframe, tasks in timeline_data.items():
    with st.expander(f"**{timeframe}**"):
        for task in tasks:
            st.markdown(f"- {task}")

# Savings Calculator
st.subheader("💰 Savings Calculator")

col1, col2 = st.columns(2)

with col1:
    implement_weeks = st.slider("Implementation Timeline (weeks)", 4, 52, 12)

with col2:
    investment = st.number_input("Initial Investment (₹)", 0, 100000, 10000, step=1000)

if monthly_cost > 0:
    monthly_savings = potential_savings
    annual_savings = monthly_savings * 12
    roi_months = (investment / monthly_savings) if monthly_savings > 0 else 0
    
    st.markdown(f"""
    **Financial Analysis:**
    - **Monthly Savings:** ₹{monthly_savings:,.0f}
    - **Annual Savings:** ₹{annual_savings:,.0f}
    - **ROI Period:** {roi_months:.1f} months
    - **5-Year Savings:** ₹{annual_savings * 5:,.0f}
    """)

# Action Plan
st.divider()
st.subheader("📋 Your Action Plan")

if st.button("📥 Download Action Plan", type="primary", use_container_width=True):
    st.success("Action plan generated! (Simulated for demo)")

if st.button("🔄 Get Updated Recommendations", use_container_width=True):
    st.rerun()

# Footer
st.divider()
st.caption("💡 Tips: Start with high-priority, low-investment items for quick wins")

