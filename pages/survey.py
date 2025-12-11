import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import time

st.set_page_config(page_title="Energy Survey", page_icon="📋", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .appliance-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Smart Energy Consumption Survey")
st.caption("Help us understand your energy usage patterns for accurate forecasting")

# Initialize session state
if "survey_step" not in st.session_state:
    st.session_state.survey_step = 0
if "survey_completed" not in st.session_state:
    st.session_state.survey_completed = False
if "survey_data" not in st.session_state:
    st.session_state.survey_data = {}

# Progress Bar
steps = ["🏠 Household Info", "🔌 Appliances", "🕒 Usage Patterns", "📊 Review"]
progress_value = (st.session_state.survey_step + 1) / len(steps)
st.progress(progress_value, text=f"Step {st.session_state.survey_step + 1} of {len(steps)}")

# Step Navigation
cols = st.columns(len(steps))
for i, step in enumerate(steps):
    with cols[i]:
        if i == st.session_state.survey_step:
            st.markdown(f"**{step.split()[1]}** 🟢")
        elif i < st.session_state.survey_step:
            st.markdown(f"{step.split()[1]} ✅")
        else:
            st.markdown(f"{step.split()[1]} ⚪")

st.divider()

# Step 1: Household Info
if st.session_state.survey_step == 0:
    st.subheader("🏠 Household Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        members = st.number_input("👨‍👩‍👧‍👦 Number of Family Members", 1, 20, 4, 
                                help="Include all permanent residents")
        house_type = st.selectbox("🏡 Type of Residence", 
                                 ["Apartment (1-2 BHK)", "Apartment (3+ BHK)", 
                                  "Independent House", "Villa", "Duplex"])
        bedrooms = st.number_input("🛏️ Number of Bedrooms", 1, 10, 3)
    
    with col2:
        area = st.slider("📏 Total Area (Square Feet)", 300, 10000, 1200, 100)
        location = st.selectbox("📍 City/Region", 
                               ["Chennai", "Coimbatore", "Madurai", "Trichy", 
                                "Salem", "Bangalore", "Hyderabad", "Other"])
        construction_year = st.slider("🏗️ Construction Year", 1980, 2024, 2010)
    
    # Store data
    st.session_state.survey_data["household"] = {
        "members": members,
        "type": house_type,
        "bedrooms": bedrooms,
        "area": area,
        "location": location,
        "construction_year": construction_year
    }

# Step 2: Appliances (IMPROVED VERSION)
elif st.session_state.survey_step == 1:
    st.subheader("🔌 Smart Appliance Analysis")
    
    # Interactive information
    with st.expander("💡 Understanding Your Appliances"):
        st.markdown("""
        **Why this matters:**
        - Each appliance contributes differently to your energy bill
        - Smart usage patterns can save up to 30% on electricity
        - We'll help you identify optimization opportunities
        """)
    
    # Appliances database with more details
    appliances_db = [
        {"name": "Refrigerator", "category": "Essential", "icon": "❄️", 
         "default_watts": 150, "default_hours": 24, "efficiency": ["5-star", "3-star", "Old"]},
        {"name": "Air Conditioner", "category": "Cooling", "icon": "❄️", 
         "default_watts": 1500, "default_hours": 8, "efficiency": ["Inverter", "Non-inverter"]},
        {"name": "Ceiling Fan", "category": "Cooling", "icon": "🌀", 
         "default_watts": 75, "default_hours": 12, "efficiency": ["BLDC", "Standard"]},
        {"name": "LED Lighting", "category": "Lighting", "icon": "💡", 
         "default_watts": 10, "default_hours": 6, "efficiency": ["Smart", "Standard"]},
        {"name": "Television", "category": "Entertainment", "icon": "📺", 
         "default_watts": 120, "default_hours": 4, "efficiency": ["LED", "OLED", "Plasma"]},
        {"name": "Washing Machine", "category": "Laundry", "icon": "👕", 
         "default_watts": 500, "default_hours": 1, "efficiency": ["Front Load", "Top Load"]},
        {"name": "Water Heater", "category": "Heating", "icon": "🔥", 
         "default_watts": 2000, "default_hours": 1, "efficiency": ["Instant", "Storage", "Solar"]},
        {"name": "Microwave Oven", "category": "Kitchen", "icon": "🍳", 
         "default_watts": 1200, "default_hours": 0.5, "efficiency": ["Convection", "Standard"]},
        {"name": "Air Purifier", "category": "Health", "icon": "🌬️", 
         "default_watts": 50, "default_hours": 8, "efficiency": ["HEPA", "Standard"]},
        {"name": "Water Purifier", "category": "Kitchen", "icon": "💧", 
         "default_watts": 40, "default_hours": 6, "efficiency": ["RO", "UV"]},
        {"name": "Laptop/Computer", "category": "Office", "icon": "💻", 
         "default_watts": 100, "default_hours": 6, "efficiency": ["Gaming", "Standard"]},
        {"name": "Electric Vehicle Charger", "category": "Transport", "icon": "🚗", 
         "default_watts": 7000, "default_hours": 4, "efficiency": ["Fast", "Slow"]},
    ]
    
    # Smart appliance grid
    appliance_data = []
    total_monthly_kwh = 0
    
    st.markdown("### 📋 Appliance Details")
    
    # Create a grid of appliance cards
    cols = st.columns(2)
    
    for idx, appliance in enumerate(appliances_db):
        with cols[idx % 2]:
            with st.container():
                st.markdown(f"""
                <div class='appliance-card'>
                    <h4>{appliance['icon']} {appliance['name']}</h4>
                    <small><i>{appliance['category']}</i></small>
                </div>
                """, unsafe_allow_html=True)
                
                col_qty, col_hrs = st.columns(2)
                with col_qty:
                    quantity = st.number_input(
                        "Quantity", 
                        0, 10, 1,
                        key=f"qty_{appliance['name']}",
                        help="Number of this appliance"
                    )
                
                with col_hrs:
                    hours = st.slider(
                        "Hours/day", 
                        0.0, 24.0, 
                        float(appliance['default_hours']), 0.5,
                        key=f"hours_{appliance['name']}",
                        help="Average daily usage"
                    )
                
                # Efficiency rating
                efficiency = st.selectbox(
                    "Efficiency",
                    appliance['efficiency'],
                    key=f"eff_{appliance['name']}",
                    help="Select the efficiency rating"
                )
                
                # Calculate consumption
                if quantity > 0 and hours > 0:
                    # Adjust watts based on efficiency
                    efficiency_factor = {
                        "5-star": 0.7, "Inverter": 0.6, "BLDC": 0.5, "Smart": 0.8,
                        "Front Load": 0.7, "Solar": 0.3, "Convection": 0.9, "HEPA": 0.8,
                        "RO": 0.9, "Gaming": 1.2, "Fast": 1.1
                    }.get(efficiency, 1.0)
                    
                    adjusted_watts = appliance['default_watts'] * efficiency_factor
                    monthly_kwh = (adjusted_watts * quantity * hours * 30) / 1000
                    total_monthly_kwh += monthly_kwh
                    
                    # Color code based on consumption
                    color = "🟢" if monthly_kwh < 50 else "🟡" if monthly_kwh < 150 else "🔴"
                    
                    st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                        <b>Monthly Consumption:</b> {color} {monthly_kwh:.1f} kWh
                        <br><small>Cost: ₹{monthly_kwh * 8:.0f} @ ₹8/kWh</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    appliance_data.append({
                        "name": appliance['name'],
                        "category": appliance['category'],
                        "quantity": quantity,
                        "hours": hours,
                        "efficiency": efficiency,
                        "watts": adjusted_watts,
                        "monthly_kwh": monthly_kwh,
                        "cost": monthly_kwh * 8  # Assuming ₹8 per kWh
                    })
    
    # Summary Dashboard
    st.divider()
    st.markdown("### 📈 Consumption Summary")
    
    if appliance_data:
        # Create a pie chart of consumption by category
        df_appliances = pd.DataFrame(appliance_data)
        category_sum = df_appliances.groupby('category')['monthly_kwh'].sum().reset_index()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📱 Total Appliances", len([a for a in appliance_data if a['quantity'] > 0]))
        
        with col2:
            st.metric("⚡ Monthly Consumption", f"{total_monthly_kwh:.1f} kWh")
        
        with col3:
            st.metric("💰 Estimated Cost", f"₹{total_monthly_kwh * 8:.0f}")
        
        # Visualizations
        tab1, tab2 = st.tabs(["📊 By Category", "🔝 Top Consumers"])
        
        with tab1:
            fig = px.pie(category_sum, values='monthly_kwh', names='category',
                        title='Energy Consumption by Category',
                        color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            top_consumers = df_appliances.nlargest(5, 'monthly_kwh')
            fig = px.bar(top_consumers, x='name', y='monthly_kwh',
                        title='Top 5 Energy Consuming Appliances',
                        color='monthly_kwh',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    st.session_state.survey_data["appliances"] = appliance_data
    st.session_state.survey_data["total_appliance_kwh"] = total_monthly_kwh

# Step 3: Usage Patterns
elif st.session_state.survey_step == 2:
    st.subheader("🕒 Smart Usage Pattern Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📅 Daily Usage Timeline")
        
        # Interactive timeline
        st.markdown("**Drag sliders to set your typical usage pattern:**")
        
        morning_peak = st.slider("🌅 Morning Peak (6-10 AM)", 0, 100, 30,
                                help="Percentage of daily usage during morning hours")
        day_usage = st.slider("☀️ Daytime (10 AM-4 PM)", 0, 100, 20)
        evening_peak = st.slider("🌆 Evening Peak (6-10 PM)", 0, 100, 40)
        night_usage = st.slider("🌙 Night (10 PM-6 AM)", 0, 100, 10)
        
        # Validate percentages
        total = morning_peak + day_usage + evening_peak + night_usage
        if total != 100:
            st.warning(f"⚠️ Total is {total}%. Please adjust to make it 100%.")
        
        # Create timeline visualization
        hours = ['6-10 AM', '10 AM-4 PM', '4-6 PM', '6-10 PM', '10 PM-6 AM']
        usage = [morning_peak, day_usage, 0, evening_peak, night_usage]
        
        fig = go.Figure(data=[go.Bar(x=hours, y=usage, marker_color='#1f77b4')])
        fig.update_layout(title='Daily Energy Usage Pattern', 
                         yaxis_title='Percentage (%)',
                         xaxis_title='Time of Day')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🌡️ Seasonal Factors")
        
        ac_months = st.slider("❄️ AC Usage (Months/Year)", 0, 12, 6,
                             help="Number of months you typically use air conditioning")
        
        st.markdown("#### 💧 Water Heating")
        geyser_type = st.selectbox("Water Heater Type", 
                                  ["Instant Electric", "Storage Electric", 
                                   "Solar Assisted", "Gas", "None"])
        
        if geyser_type != "None":
            geyser_hours = st.slider("Daily usage (hours)", 0.5, 4.0, 1.5, 0.5)
        
        st.markdown("#### 🍳 Cooking")
        cooking_fuel = st.multiselect("Primary Cooking Fuel",
                                     ["LPG/Cylinder", "Induction", "Microwave", 
                                      "Gas Pipeline", "Electric Stove", "Other"])
    
    st.markdown("#### 🏠 Home Automation")
    automation = st.multiselect("Smart Home Features",
                               ["Smart Lighting", "Smart Thermostat", 
                                "Energy Monitoring", "Smart Plugs", 
                                "Solar Panel Monitoring", "None"])
    
    st.session_state.survey_data["usage"] = {
        "morning_peak": morning_peak,
        "day_usage": day_usage,
        "evening_peak": evening_peak,
        "night_usage": night_usage,
        "ac_months": ac_months,
        "geyser_type": geyser_type,
        "geyser_hours": geyser_hours if geyser_type != "None" else 0,
        "cooking_fuel": cooking_fuel,
        "automation": automation
    }

# Step 4: Review
elif st.session_state.survey_step == 3:
    st.subheader("📊 Final Review & Insights")
    
    # Display Summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏠 Household Summary")
        if "household" in st.session_state.survey_data:
            household = st.session_state.survey_data["household"]
            st.markdown(f"""
            <div class='metric-card'>
                <b>👨‍👩‍👧‍👦 Family Members:</b> {household['members']}<br>
                <b>🏡 Residence:</b> {household['type']}<br>
                <b>🛏️ Bedrooms:</b> {household['bedrooms']}<br>
                <b>📏 Area:</b> {household['area']} sq.ft<br>
                <b>📍 Location:</b> {household['location']}<br>
                <b>🏗️ Built:</b> {household['construction_year']}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ⚡ Energy Summary")
        if "total_appliance_kwh" in st.session_state.survey_data:
            total_kwh = st.session_state.survey_data["total_appliance_kwh"]
            
            # Enhanced bill calculation (TNEB rates)
            if total_kwh <= 100:
                bill = 0
                unit_rate = 0
            elif total_kwh <= 200:
                bill = (total_kwh - 100) * 2.25
                unit_rate = 2.25
            elif total_kwh <= 400:
                bill = 100 * 2.25 + (total_kwh - 200) * 4.50
                unit_rate = 4.50
            else:
                bill = 100 * 2.25 + 200 * 4.50 + (total_kwh - 400) * 6.00
                unit_rate = 6.00
            
            bill += 50  # Fixed charges
            
            # CO2 emissions calculation (approx 0.85 kg CO2 per kWh in India)
            co2_emissions = total_kwh * 0.85
            
            st.markdown(f"""
            <div class='metric-card'>
                <b>📈 Monthly Consumption:</b> {total_kwh:.1f} kWh<br>
                <b>💰 Estimated Bill:</b> ₹{bill:,.0f}<br>
                <b>📊 Avg. Rate:</b> ₹{unit_rate:.2f}/kWh<br>
                <b>🌍 CO2 Emissions:</b> {co2_emissions:.1f} kg<br>
                <b>🌳 Equivalent Trees:</b> {co2_emissions/21:.0f} trees needed
            </div>
            """, unsafe_allow_html=True)
    
    # Savings Recommendations
    st.divider()
    st.markdown("### 💡 Personalized Recommendations")
    
    if "total_appliance_kwh" in st.session_state.survey_data:
        total_kwh = st.session_state.survey_data["total_appliance_kwh"]
        
        recommendations = []
        
        if total_kwh > 500:
            recommendations.append("🔴 **High Consumption Alert:** Consider energy audit")
        
        if "appliances" in st.session_state.survey_data:
            old_appliances = [a for a in st.session_state.survey_data["appliances"] 
                            if "Old" in str(a.get('efficiency', ''))]
            if old_appliances:
                recommendations.append("🔄 **Upgrade Opportunity:** Replace old appliances with 5-star rated ones")
        
        if st.session_state.survey_data.get("usage", {}).get("evening_peak", 0) > 50:
            recommendations.append("⏰ **Peak Shifting:** Shift some usage to off-peak hours (10 PM - 6 AM)")
        
        if len(recommendations) > 0:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("✅ Your energy profile looks efficient! Keep up the good work.")
    
    st.divider()
    
    # Final Confirmation
    st.markdown("### ✅ Final Step")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        agree = st.checkbox("I confirm all information provided is accurate to the best of my knowledge",
                           help="Your data will be used for personalized forecasting")
    
    with col2:
        if agree:
            if st.button("🚀 Complete Survey", type="primary", use_container_width=True):
                # Store all data in session state
                st.session_state.survey_completed = True
                st.session_state.user_data = {
                    "monthly_consumption": st.session_state.survey_data.get("total_appliance_kwh", 0),
                    "monthly_cost": bill if 'bill' in locals() else 0,
                    "survey_timestamp": pd.Timestamp.now(),
                    **st.session_state.survey_data
                }
                st.success("🎉 Survey completed successfully!")
                st.balloons()
                st.rerun()

# Navigation Buttons (always visible)
st.divider()

# Show completion options if survey is completed
if st.session_state.survey_completed:
    st.markdown("### 🎉 What would you like to do next?")


    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Go to Data Analysis", use_container_width=True, icon="📈"):
            st.switch_page("pages/data_loader.py")
    
    with col2:
        if st.button("🤖 AI Forecasting", use_container_width=True, icon="🔮"):
            st.switch_page("pages/forecast.py")
    
    with col3:
        if st.button("🏠 Back to Dashboard", use_container_width=True, icon="🏠"):
            st.switch_page("main.py")
    
    # Show a summary card
    if "user_data" in st.session_state:
        st.info(f"""
        **Survey Summary:** 
        - Monthly Consumption: {st.session_state.user_data.get('monthly_consumption', 0):.1f} kWh
        - Estimated Cost: ₹{st.session_state.user_data.get('monthly_cost', 0):,.0f}
        - Appliances Logged: {len(st.session_state.user_data.get('appliances', []))}
        """)
else:
 
st.divider()

# Redo Survey and Navigation Options Section
st.markdown("### 🔄 Survey Options")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("↩️ Go Back to Survey", use_container_width=True):
        # Go back to the last step
        if st.session_state.survey_step > 0:
            st.session_state.survey_step -= 1
        st.rerun()

with col2:
    if st.button("🔄 Start Over", use_container_width=True, type="secondary"):
        # Reset the entire survey
        st.session_state.survey_step = 0
        st.session_state.survey_completed = False
        st.session_state.survey_data = {}
        if 'user_data' in st.session_state:
            del st.session_state.user_data
        st.rerun()

with col3:
    if st.button("📥 Export Data", use_container_width=True, type="secondary"):
        # Export survey data as JSON
        import json
        survey_json = json.dumps(st.session_state.user_data, indent=2)
        st.download_button(
            label="Download Survey Data (JSON)",
            data=survey_json,
            file_name="energy_survey_data.json",
            mime="application/json",
            use_container_width=True
        )

# Add a more prominent Previous button during survey steps
if not st.session_state.survey_completed and st.session_state.survey_step > 0:
    st.divider()
    if st.button("⬅️ Go Back to Previous Step", use_container_width=True, icon="↩️"):
        st.session_state.survey_step -= 1
        st.rerun()
    
