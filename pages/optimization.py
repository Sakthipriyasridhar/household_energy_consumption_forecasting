import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Optimization", page_icon="💡", layout="wide")
st.title("💡 Smart Energy Optimization")

# Custom CSS for better styling
st.markdown("""
<style>
    .priority-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .priority-medium {
        background: linear-gradient(135deg, #ffd166, #ffb347);
        color: #333;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .priority-low {
        background: linear-gradient(135deg, #06d6a0, #1b9aaa);
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .recommendation-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #2E86AB;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s;
    }
    .recommendation-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    .savings-badge {
        background: linear-gradient(135deg, #73AB84, #4CAF50);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1em;
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
    
    # Show sample recommendations with interactive features
    st.divider()
    st.subheader("📋 Sample Recommendations Preview")
    
    # Interactive demo with filters
    demo_tab1, demo_tab2 = st.tabs(["🎯 By Priority", "💰 By Savings"])
    
    with demo_tab1:
        priority_filter = st.radio("Filter by Priority:", ["All", "High", "Medium", "Low"], horizontal=True)
    
    with demo_tab2:
        savings_min, savings_max = st.slider("Monthly Savings Range (₹)", 100, 1000, (200, 600))
    
    sample_recommendations = [
        {
            "appliance": "Air Conditioner",
            "current": "₹800/month",
            "optimized": "₹500/month",
            "savings": "₹300/month",
            "action": "Set temperature to 24°C, use timer, clean filters monthly",
            "priority": "High",
            "payback": "Immediate",
            "difficulty": "Easy",
            "icon": "❄️"
        },
        {
            "appliance": "Refrigerator (10+ years)",
            "current": "₹600/month",
            "optimized": "₹300/month",
            "savings": "₹300/month",
            "action": "Replace with 5-star rated model (BEE rating)",
            "priority": "Medium",
            "payback": "2-3 years",
            "difficulty": "Medium",
            "icon": "🧊"
        },
        {
            "appliance": "Lighting",
            "current": "₹400/month",
            "optimized": "₹150/month",
            "savings": "₹250/month",
            "action": "Switch all bulbs to LED (10W vs 60W)",
            "priority": "Low",
            "payback": "6 months",
            "difficulty": "Easy",
            "icon": "💡"
        },
        {
            "appliance": "Water Heater",
            "current": "₹500/month",
            "optimized": "₹250/month",
            "savings": "₹250/month",
            "action": "Install timer, reduce temperature to 50°C",
            "priority": "Medium",
            "payback": "1 year",
            "difficulty": "Medium",
            "icon": "🔥"
        }
    ]
    
    # Filter sample data
    filtered_samples = []
    for rec in sample_recommendations:
        savings_value = int(rec["savings"].replace("₹", "").replace("/month", ""))
        if priority_filter != "All" and rec["priority"] != priority_filter:
            continue
        if savings_value < savings_min or savings_value > savings_max:
            continue
        filtered_samples.append(rec)
    
    # Display filtered samples
    for rec in filtered_samples:
        with st.container():
            col_left, col_right = st.columns([3, 1])
            
            with col_left:
                st.markdown(f"""
                <div class='recommendation-card'>
                    <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                        <span style='font-size: 2em; margin-right: 15px;'>{rec['icon']}</span>
                        <div>
                            <h4 style='margin: 0; color: #2E86AB;'>{rec['appliance']}</h4>
                            <p style='margin: 5px 0; font-size: 0.9em; color: #666;'>
                                <strong>Current:</strong> {rec['current']} → <strong>Optimized:</strong> {rec['optimized']}
                            </p>
                        </div>
                    </div>
                    <p style='margin: 10px 0;'><strong>📌 Action:</strong> {rec['action']}</p>
                    <div style='display: flex; gap: 10px; margin-top: 15px;'>
                        <span class='priority-{rec["priority"].lower()}'>{rec['priority']} Priority</span>
                        <span style='background: #e3f2fd; color: #1976d2; padding: 3px 10px; border-radius: 15px; font-size: 0.8em;'>
                            Payback: {rec['payback']}
                        </span>
                        <span style='background: #f3e5f5; color: #7b1fa2; padding: 3px 10px; border-radius: 15px; font-size: 0.8em;'>
                            Difficulty: {rec['difficulty']}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_right:
                st.markdown(f"""
                <div style='text-align: center; padding: 15px;'>
                    <div class='savings-badge'>{rec['savings']}</div>
                    <p style='font-size: 0.9em; color: #666; margin-top: 10px;'>Monthly Savings</p>
                </div>
                """, unsafe_allow_html=True)
    
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
    st.metric(
        "Current Monthly Cost",
        f"₹{monthly_cost:,.0f}",
        delta=None,
        delta_color="normal"
    )

with col2:
    st.metric(
        "Potential Monthly Savings",
        f"₹{potential_savings:,.0f}",
        delta=f"{35}%",
        delta_color="inverse"
    )

with col3:
    savings_percent = (potential_savings / monthly_cost * 100) if monthly_cost > 0 else 0
    st.metric(
        "Optimized Cost",
        f"₹{optimized_cost:,.0f}",
        delta=f"-{savings_percent:.0f}%",
        delta_color="inverse"
    )

with col4:
    annual_savings = potential_savings * 12
    st.metric(
        "Annual Savings Potential",
        f"₹{annual_savings:,.0f}",
        delta="Projected",
        delta_color="off"
    )

# Savings Visualization
st.markdown("#### 📈 Savings Breakdown")
fig = go.Figure()

fig.add_trace(go.Bar(
    name='Current Cost',
    x=['Monthly Cost'],
    y=[monthly_cost],
    marker_color='#2E86AB',
    text=[f'₹{monthly_cost:,.0f}'],
    textposition='auto',
))

fig.add_trace(go.Bar(
    name='Potential Savings',
    x=['Monthly Cost'],
    y=[potential_savings],
    marker_color='#73AB84',
    text=[f'₹{potential_savings:,.0f}'],
    textposition='auto',
))

fig.update_layout(
    title='Current Cost vs Potential Savings',
    barmode='stack',
    height=300,
    showlegend=True,
    yaxis_title='Amount (₹)'
)

st.plotly_chart(fig, use_container_width=True)

# Personalized Recommendations with Advanced Logic
st.subheader("🎯 Smart Recommendations Engine")

# Generate recommendations based on user data
recommendations = []

# Analyze appliances
ac_count = 0
old_appliances = []
led_percentage = 0

for appliance in appliances:
    name = appliance.get("name", "").lower()
    quantity = appliance.get("quantity", 0)
    age = appliance.get("age", 0)
    hours = appliance.get("hours", 0)
    
    # AC Analysis
    if "air" in name or "ac" in name:
        ac_count += quantity
        if hours > 8:
            recommendations.append({
                "title": "Air Conditioner Optimization",
                "description": f"Your AC runs {hours} hours/day - above optimal",
                "action": "Reduce to 6-8 hours, set temp to 24°C, use sleep mode, clean filters",
                "savings": f"₹{min(800, hours * 50)}/month",
                "priority": "High",
                "icon": "❄️",
                "category": "Cooling",
                "investment": "Low (₹0-2,000)",
                "payback": "Immediate",
                "co2_reduction": "50-100 kg/month"
            })
    
    # Refrigerator Analysis
    if "refrigerator" in name or "fridge" in name:
        if age > 8:
            old_appliances.append("Refrigerator")
            recommendations.append({
                "title": "Refrigerator Upgrade",
                "description": f"{age}-year old model - inefficient",
                "action": "Replace with 5-star BEE rated refrigerator",
                "savings": "₹400-600/month",
                "priority": "Medium",
                "icon": "🧊",
                "category": "Kitchen",
                "investment": "High (₹20,000-40,000)",
                "payback": "3-4 years",
                "co2_reduction": "100-150 kg/month"
            })
    
    # Lighting Analysis
    if "light" in name:
        total_bulbs = quantity
        if "led" not in name and "tube" not in name:
            led_percentage = 0
            recommendations.append({
                "title": "Complete LED Conversion",
                "description": f"You have {total_bulbs} non-LED bulbs",
                "action": "Replace all with 9W LED bulbs",
                "savings": f"₹{total_bulbs * 30}/month",
                "priority": "Low",
                "icon": "💡",
                "category": "Lighting",
                "investment": f"Medium (₹{total_bulbs * 200})",
                "payback": "6-9 months",
                "co2_reduction": f"{total_bulbs * 5} kg/month"
            })

# Add smart recommendations
if monthly_consumption > 500:
    recommendations.append({
        "title": "Peak Load Management",
        "description": "High consumption detected during peak hours",
        "action": "Install smart plugs, shift heavy loads to off-peak (10 PM - 6 AM)",
        "savings": "₹200-400/month",
        "priority": "Medium",
        "icon": "🕒",
        "category": "Smart Home",
        "investment": "Medium (₹3,000-5,000)",
        "payback": "1-1.5 years",
        "co2_reduction": "30-60 kg/month"
    })

if location in ["Chennai", "Madurai", "Coimbatore"]:
    recommendations.append({
        "title": "Solar Water Heater",
        "description": f"Excellent solar potential in {location}",
        "action": "Install 100L solar water heater",
        "savings": "₹300-500/month",
        "priority": "Medium",
        "icon": "☀️",
        "category": "Renewable",
        "investment": "High (₹25,000-35,000)",
        "payback": "4-5 years",
        "co2_reduction": "80-120 kg/month"
    })

# Add behavioral recommendations
recommendations.extend([
    {
        "title": "Standby Power Elimination",
        "description": "Phantom load from devices on standby",
        "action": "Use power strips with switches, turn off completely when not in use",
        "savings": "₹100-200/month",
        "priority": "Low",
        "icon": "🔌",
        "category": "Behavioral",
        "investment": "Low (₹500-1,000)",
        "payback": "3-5 months",
        "co2_reduction": "15-30 kg/month"
    },
    {
        "title": "Natural Ventilation",
        "description": "Reduce AC dependency",
        "action": "Use cross-ventilation, ceiling fans strategically",
        "savings": "₹150-250/month",
        "priority": "Low",
        "icon": "💨",
        "category": "Behavioral",
        "investment": "Low (₹0-500)",
        "payback": "Immediate",
        "co2_reduction": "20-40 kg/month"
    }
])

# Filter and Sort Options
st.markdown("#### 🔍 Filter Recommendations")

col_filter1, col_filter2, col_filter3 = st.columns(3)
with col_filter1:
    priority_filter = st.multiselect(
        "Priority Level",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )
with col_filter2:
    category_filter = st.multiselect(
        "Category",
        sorted(list(set([r["category"] for r in recommendations]))),
        default=sorted(list(set([r["category"] for r in recommendations])))
    )
with col_filter3:
    sort_by = st.selectbox(
        "Sort by",
        ["Priority (High to Low)", "Savings (High to Low)", "Payback (Fast to Slow)"]
    )

# Filter recommendations
filtered_recommendations = []
for rec in recommendations:
    if rec["priority"] in priority_filter and rec["category"] in category_filter:
        filtered_recommendations.append(rec)

# Sort recommendations
if sort_by == "Priority (High to Low)":
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    filtered_recommendations.sort(key=lambda x: priority_order[x["priority"]])
elif sort_by == "Savings (High to Low)":
    filtered_recommendations.sort(key=lambda x: int(x["savings"].replace("₹", "").split("-")[0]), reverse=True)
elif sort_by == "Payback (Fast to Slow)":
    payback_order = {"Immediate": 0, "3-5 months": 1, "6-9 months": 2, "1-1.5 years": 3, "3-4 years": 4, "4-5 years": 5}
    filtered_recommendations.sort(key=lambda x: payback_order.get(x["payback"], 99))

# Display filtered recommendations
if not filtered_recommendations:
    st.info("No recommendations match your filters. Try adjusting filter settings.")
else:
    for rec in filtered_recommendations:
        with st.container():
            st.markdown(f"""
            <div class='recommendation-card'>
                <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;'>
                    <div style='display: flex; align-items: center; gap: 15px;'>
                        <span style='font-size: 2.5em;'>{rec['icon']}</span>
                        <div>
                            <h4 style='margin: 0; color: #2E86AB;'>{rec['title']}</h4>
                            <p style='margin: 5px 0; color: #666; font-size: 0.9em;'>
                                <span style='background: #f0f0f0; padding: 2px 8px; border-radius: 10px; margin-right: 10px;'>
                                    {rec['category']}
                                </span>
                                <span class='priority-{rec["priority"].lower()}'>{rec['priority']} Priority</span>
                            </p>
                        </div>
                    </div>
                    <div style='text-align: right;'>
                        <div class='savings-badge'>{rec['savings']}</div>
                        <p style='margin: 5px 0; font-size: 0.8em; color: #666;'>
                            CO₂ Reduction: {rec['co2_reduction']}
                        </p>
                    </div>
                </div>
                
                <p style='margin: 10px 0; color: #555;'>{rec['description']}</p>
                
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;'>
                    <strong style='color: #2E86AB;'>📌 Action Plan:</strong> {rec['action']}
                </div>
                
                <div style='display: flex; justify-content: space-between; margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee;'>
                    <div>
                        <strong>Investment Required:</strong> {rec['investment']}
                    </div>
                    <div>
                        <strong>Payback Period:</strong> {rec['payback']}
                    </div>
                    <div>
                        <button style='
                            background: linear-gradient(135deg, #2E86AB, #4CAF50);
                            color: white;
                            border: none;
                            padding: 8px 20px;
                            border-radius: 20px;
                            cursor: pointer;
                            font-weight: bold;
                        '>➕ Add to Action Plan</button>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Implementation Timeline with Gantt Chart
st.subheader("📅 Smart Implementation Timeline")

# Create timeline data
timeline_phases = {
    "Immediate (Week 1-2)": [
        {"task": "Set AC temperature to 24°C", "duration": 1, "owner": "You"},
        {"task": "Replace 5 highest-use bulbs with LED", "duration": 2, "owner": "You"},
        {"task": "Install power strips for standby devices", "duration": 3, "owner": "Electrician"}
    ],
    "Short-term (Month 1)": [
        {"task": "Complete LED lighting conversion", "duration": 7, "owner": "You"},
        {"task": "Install smart plugs for heavy appliances", "duration": 14, "owner": "Electrician"},
        {"task": "Implement off-peak scheduling", "duration": 3, "owner": "You"}
    ],
    "Medium-term (Month 2-3)": [
        {"task": "Refrigerator energy audit", "duration": 30, "owner": "Professional"},
        {"task": "Solar water heater assessment", "duration": 45, "owner": "Vendor"},
        {"task": "Whole-house energy monitoring", "duration": 60, "owner": "Professional"}
    ],
    "Long-term (Month 4-6)": [
        {"task": "Major appliance upgrades", "duration": 90, "owner": "Vendor"},
        {"task": "Solar PV system evaluation", "duration": 120, "owner": "Solar Company"},
        {"task": "Home automation integration", "duration": 180, "owner": "Smart Home Expert"}
    ]
}

# Display timeline as expandable sections
for phase, tasks in timeline_phases.items():
    with st.expander(f"**{phase}** - Estimated Savings: ₹{len(tasks)*150}/month", expanded=True if phase == "Immediate (Week 1-2)" else False):
        for task in tasks:
            col_task1, col_task2, col_task3 = st.columns([3, 1, 1])
            with col_task1:
                st.markdown(f"✅ **{task['task']}**")
            with col_task2:
                st.markdown(f"⏱️ {task['duration']} day(s)")
            with col_task3:
                st.markdown(f"👤 {task['owner']}")

# Advanced Savings Calculator
st.subheader("💰 Advanced Savings Calculator")

calc_col1, calc_col2 = st.columns(2)

with calc_col1:
    selected_recommendations = st.multiselect(
        "Select recommendations to implement:",
        [f"{r['title']} ({r['savings']})" for r in recommendations],
        default=[f"{r['title']} ({r['savings']})" for r in recommendations if r["priority"] == "High"]
    )
    
    implementation_months = st.slider(
        "Implementation timeline (months):",
        1, 24, 6,
        help="How long will it take to implement all selected recommendations?"
    )

with calc_col2:
    investment_budget = st.number_input(
        "Your investment budget (₹):",
        0, 500000, 50000,
        step=5000,
        help="Total amount you're willing to invest in energy efficiency"
    )
    
    consider_subsidy = st.checkbox(
        "Include government subsidies (up to 40%)",
        value=True,
        help="Many energy efficiency upgrades qualify for government subsidies"
    )

# Calculate results
if selected_recommendations and monthly_cost > 0:
    # Estimate savings from selected recommendations
    total_monthly_savings = 0
    total_investment = 0
    
    for rec_title in selected_recommendations:
        for rec in recommendations:
            if rec["title"] in rec_title:
                # Extract savings amount
                savings_str = rec["savings"].replace("₹", "").replace("/month", "")
                if "-" in savings_str:
                    min_sav, max_sav = map(int, savings_str.split("-"))
                    avg_savings = (min_sav + max_sav) / 2
                else:
                    avg_savings = int(savings_str)
                
                # Extract investment amount
                inv_str = rec["investment"]
                if "₹" in inv_str:
                    inv_value = int(''.join(filter(str.isdigit, inv_str)))
                else:
                    inv_value = 0
                
                total_monthly_savings += avg_savings
                total_investment += inv_value
    
    # Apply subsidy if selected
    if consider_subsidy:
        subsidy_amount = total_investment * 0.4
        net_investment = total_investment - subsidy_amount
    else:
        subsidy_amount = 0
        net_investment = total_investment
    
    # Calculate ROI
    annual_savings = total_monthly_savings * 12
    if annual_savings > 0:
        roi_years = net_investment / annual_savings
    else:
        roi_years = 0
    
    # Display results
    st.markdown("#### 📊 Investment Analysis")
    
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.metric("Total Monthly Savings", f"₹{total_monthly_savings:,.0f}")
    
    with result_col2:
        st.metric("Net Investment", f"₹{net_investment:,.0f}")
    
    with result_col3:
        st.metric("ROI Period", f"{roi_years:.1f} years")
    
    # Progress visualization
    st.markdown("#### 📈 Budget Utilization")
    
    budget_used = min(net_investment, investment_budget)
    budget_percentage = (budget_used / investment_budget * 100) if investment_budget > 0 else 0
    
    st.progress(min(budget_percentage / 100, 1.0))
    st.caption(f"Budget used: ₹{budget_used:,.0f} of ₹{investment_budget:,.0f} ({budget_percentage:.1f}%)")
    
    if subsidy_amount > 0:
        st.info(f"✅ You qualify for approximately ₹{subsidy_amount:,.0f} in government subsidies!")

# Action Plan Generator
st.divider()
st.subheader("📋 Generate Your Custom Action Plan")

plan_col1, plan_col2 = st.columns([2, 1])

with plan_col1:
    plan_name = st.text_input("Action Plan Name:", "My Energy Efficiency Plan")
    
    include_timeline = st.checkbox("Include detailed timeline", value=True)
    include_budget = st.checkbox("Include budget breakdown", value=True)
    include_vendors = st.checkbox("Include vendor recommendations", value=True)

with plan_col2:
    st.markdown("###")
    if st.button("📥 Generate & Download Plan", type="primary", use_container_width=True):
        st.success("✅ Action plan generated successfully!")
        st.balloons()
        
        # Simulated download (would generate PDF in production)
        st.markdown("""
        **Your plan includes:**
        1. Priority-wise implementation checklist
        2. Detailed cost-benefit analysis
        3. Vendor contacts and quotes
        4. Government subsidy application forms
        5. Monthly tracking sheet
        
        *In a production app, this would generate a downloadable PDF report.*
        """)

# Vendor Recommendations
if include_vendors:
    st.subheader("🤝 Recommended Vendors & Partners")
    
    vendors = [
        {"name": "Tata Power Solar", "services": ["Solar", "LED", "Smart Home"], "rating": "4.8/5"},
        {"name": "Havells Energy Solutions", "services": ["Lighting", "Fans", "Wiring"], "rating": "4.6/5"},
        {"name": "LG Smart Home", "services": ["AC", "Refrigerators", "Appliances"], "rating": "4.7/5"},
        {"name": "Local BEE-certified Electrician", "services": ["Installation", "Audit", "Maintenance"], "rating": "4.5/5"},
    ]
    
    for vendor in vendors:
        with st.container():
            st.markdown(f"""
            <div style='
                background: white;
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
                border: 1px solid #e0e0e0;
            '>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <h5 style='margin: 0; color: #2E86AB;'>{vendor['name']}</h5>
                        <p style='margin: 5px 0; color: #666;'>
                            Services: {', '.join(vendor['services'])}
                        </p>
                    </div>
                    <div style='text-align: right;'>
                        <span style='background: #FFD166; color: #333; padding: 3px 10px; border-radius: 15px;'>
                            ⭐ {vendor['rating']}
                        </span>
                        <button style='
                            background: #2E86AB;
                            color: white;
                            border: none;
                            padding: 5px 15px;
                            border-radius: 15px;
                            margin-left: 10px;
                            cursor: pointer;
                        '>Contact</button>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Monthly Tracking
st.subheader("📱 Monthly Progress Tracker")

track_col1, track_col2, track_col3 = st.columns(3)

with track_col1:
    current_month = datetime.now().strftime("%B")
    st.markdown(f"### {current_month}")
    implemented = st.number_input("Recommendations implemented:", 0, len(recommendations), 2)
    actual_savings = st.number_input(f"Actual savings this month (₹):", 0, 10000, 1500)

with track_col2:
    target_savings = potential_savings
    progress = (actual_savings / target_savings * 100) if target_savings > 0 else 0
    st.metric("Target vs Actual", f"₹{actual_savings:,.0f}", delta=f"{progress:.0f}% of target")

with track_col3:
    co2_saved = actual_savings / 8 * 0.85  # Estimate CO2 savings
    trees_saved = co2_saved / 21
    st.metric("Environmental Impact", f"{co2_saved:.0f} kg CO₂", delta=f"≈ {trees_saved:.0f} trees")

# Footer with tips
st.divider()
st.markdown("""
### 💡 Pro Tips for Maximum Savings

1. **Start Quick Wins First** - Implement zero-cost behavioral changes immediately
2. **Leverage Subsidies** - Check TNERC and MNRE websites for current subsidy programs
3. **Monitor Regularly** - Use smart meters or energy monitoring apps to track progress
4. **Seasonal Adjustments** - Adjust strategies based on summer/winter consumption patterns
5. **Community Learning** - Join local energy efficiency groups for tips and support

*Remember: Energy efficiency is a journey, not a destination. Small, consistent changes yield big results over time!*
""")

# Refresh button
if st.button("🔄 Refresh Recommendations", use_container_width=True):
    st.rerun()
