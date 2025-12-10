import streamlit as st
import pandas as pd

st.set_page_config(page_title="Energy Survey", page_icon="📋")
st.title("📋 Energy Consumption Survey")

# Initialize session state
if "survey_step" not in st.session_state:
    st.session_state.survey_step = 0
if "survey_data" not in st.session_state:
    st.session_state.survey_data = {}

# Progress Bar
steps = ["Household Info", "Appliances", "Usage Patterns", "Review"]
progress = st.progress((st.session_state.survey_step + 1) / len(steps))

# Step Navigation
cols = st.columns(len(steps))
for i, step in enumerate(steps):
    with cols[i]:
        if i == st.session_state.survey_step:
            st.markdown(f"**{step}** 🟢")
        elif i < st.session_state.survey_step:
            st.markdown(f"{step} ✅")
        else:
            st.markdown(f"{step} ⚪")

st.divider()

# Step 1: Household Info
if st.session_state.survey_step == 0:
    st.subheader("🏠 Household Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        members = st.number_input("Number of Family Members", 1, 20, 4)
        house_type = st.selectbox("Type of Residence", 
                                 ["Apartment", "Independent House", "Villa", "Duplex"])
    
    with col2:
        area = st.number_input("Total Area (Square Feet)", 300, 10000, 1200)
        location = st.selectbox("City/Region", 
                               ["Chennai", "Coimbatore", "Madurai", "Trichy", "Salem", "Other"])
    
    # Store data
    st.session_state.survey_data["household"] = {
        "members": members,
        "type": house_type,
        "area": area,
        "location": location
    }

# Step 2: Appliances
elif st.session_state.survey_step == 1:
    st.subheader("🔌 Household Appliances")
    st.markdown("Select appliances and specify usage")
    
    appliances = [
        {"name": "Refrigerator", "default_watts": 150, "default_hours": 24},
        {"name": "Air Conditioner", "default_watts": 1500, "default_hours": 8},
        {"name": "Fan", "default_watts": 75, "default_hours": 12},
        {"name": "LED Lights", "default_watts": 10, "default_hours": 6},
        {"name": "Television", "default_watts": 120, "default_hours": 4},
        {"name": "Washing Machine", "default_watts": 500, "default_hours": 1},
        {"name": "Water Heater", "default_watts": 2000, "default_hours": 1},
        {"name": "Microwave", "default_watts": 1200, "default_hours": 0.5},
    ]
    
    appliance_data = []
    
    for appliance in appliances:
        with st.expander(f"**{appliance['name']}**"):
            col1, col2, col3 = st.columns(3)
            with col1:
                quantity = st.number_input(f"Quantity", 0, 10, 1, 
                                         key=f"qty_{appliance['name']}")
            with col2:
                hours = st.slider(f"Hours per day", 0.0, 24.0, 
                                float(appliance['default_hours']), 0.5,
                                key=f"hours_{appliance['name']}")
            with col3:
                if quantity > 0 and hours > 0:
                    monthly_kwh = (appliance['default_watts'] * quantity * hours * 30) / 1000
                    st.metric("Monthly kWh", f"{monthly_kwh:.1f}")
                    appliance_data.append({
                        "name": appliance['name'],
                        "quantity": quantity,
                        "hours": hours,
                        "kwh": monthly_kwh
                    })
    
    st.session_state.survey_data["appliances"] = appliance_data

# Step 3: Usage Patterns
elif st.session_state.survey_step == 2:
    st.subheader("🕒 Usage Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⏰ Peak Usage")
        peak_morning = st.slider("Morning Peak (6-10 AM) %", 0, 100, 30)
        peak_evening = st.slider("Evening Peak (6-10 PM) %", 0, 100, 50)
        
        st.markdown("#### ❄️ Cooling")
        ac_months = st.slider("AC Usage (Months/Year)", 0, 12, 6)
    
    with col2:
        st.markdown("#### 💧 Hot Water")
        geyser_type = st.selectbox("Water Heater Type", 
                                  ["Instant", "Storage", "Solar", "None"])
        
        st.markdown("#### 🍳 Cooking")
        cooking_fuel = st.multiselect("Primary Cooking Fuel",
                                     ["LPG", "Electric", "Induction", "Other"])
    
    st.session_state.survey_data["usage"] = {
        "peak_morning": peak_morning,
        "peak_evening": peak_evening,
        "ac_months": ac_months,
        "geyser_type": geyser_type,
        "cooking_fuel": cooking_fuel
    }

# Step 4: Review
elif st.session_state.survey_step == 3:
    st.subheader("📊 Review & Calculate")
    
    # Display Summary
    st.markdown("#### 🏠 Household Summary")
    if "household" in st.session_state.survey_data:
        household = st.session_state.survey_data["household"]
        st.write(f"**Members:** {household['members']}")
        st.write(f"**Residence:** {household['type']}")
        st.write(f"**Area:** {household['area']} sq.ft")
        st.write(f"**Location:** {household['location']}")
    
    # Calculate Total Consumption
    if "appliances" in st.session_state.survey_data:
        total_kwh = sum(item["kwh"] for item in st.session_state.survey_data["appliances"])
        
        # Simple bill calculation (TNEB rates simplified)
        if total_kwh <= 100:
            bill = 0
        elif total_kwh <= 200:
            bill = (total_kwh - 100) * 2.25
        elif total_kwh <= 400:
            bill = 100 * 2.25 + (total_kwh - 200) * 4.50
        else:
            bill = 100 * 2.25 + 200 * 4.50 + (total_kwh - 400) * 6.00
        
        bill += 50  # Fixed charges
        
        st.markdown("#### ⚡ Consumption Summary")
        st.metric("Total Monthly Consumption", f"{total_kwh:.1f} kWh")
        st.metric("Estimated Monthly Bill", f"₹{bill:,.0f}")
        
        # Store in main session state
        st.session_state.user_data = {
            "monthly_consumption": total_kwh,
            "monthly_cost": bill,
            **st.session_state.survey_data
        }
    
    st.divider()
    
    # Final Confirmation
    agree = st.checkbox("✅ I confirm this information is accurate")
    
    if agree:
        if st.button("Complete Survey", type="primary", use_container_width=True):
            st.session_state.survey_completed = True
            st.success("✅ Survey completed successfully!")
            st.balloons()
            st.info("Now proceed to **AI Forecast** for predictions!")

# Navigation Buttons
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.session_state.survey_step > 0:
        if st.button("⬅️ Previous"):
            st.session_state.survey_step -= 1
            st.rerun()

with col3:
    if st.session_state.survey_step < len(steps) - 1:
        if st.button("Next ➡️", type="primary"):
            st.session_state.survey_step += 1
            st.rerun()
    elif st.session_state.survey_step == len(steps) - 1:
        if st.button("Finish ✅", type="primary"):
            st.session_state.survey_completed = True
            st.success("Survey completed!")
            st.rerun()

# Update progress
progress.progress((st.session_state.survey_step + 1) / len(steps))

# At the end of survey.py, instead of automatic redirect:
if st.button("✅ Complete Survey & Continue", type="primary"):
    st.session_state.survey_completed = True
    st.session_state.user_data = survey_data  # Store survey responses
    
    st.success("✅ Survey completed successfully!")
    
    # Show options instead of forcing redirect
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Go to Forecasting", use_container_width=True):
            st.switch_page("pages/data_loader.py")
    
    with col2:
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            st.switch_page("main.py")
    
    # Don't automatically redirect
    # st.switch_page("pages/4_Forecast.py")  # REMOVE THIS LINE
