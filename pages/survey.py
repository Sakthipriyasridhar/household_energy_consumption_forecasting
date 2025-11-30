import streamlit as st
import calendar
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Energy Survey Calculator",
    page_icon="📋",
    layout="wide"
)

# Navigation
col_nav, col_title = st.columns([1, 5])
with col_nav:
    if st.button("← Back to Main"):
        st.switch_page("main.py")
with col_title:
    st.title("📋 Energy Survey & Bill Calculator")

st.markdown("Calculate your current electricity bill based on appliance usage patterns")

# Configuration
POWER_RATINGS_MAIN = {
    "Lights": 50, "Fans": 60, "AC": 1500, "Refrigerator": 150, "TV": 90, "Geyser": 2000
}

SEASON_APPLIANCES = {
    "Summer": ["Lights", "Fans", "AC", "Refrigerator", "TV"],
    "Monsoon": ["Lights", "Fans", "AC", "Refrigerator", "TV"],
    "Winter": ["Lights", "Fans", "Refrigerator", "TV", "Geyser"]
}

SEASON_DEFAULT_USAGE = {
    "Summer": {"Lights": 6, "Fans": 10, "AC": 8, "TV": 3, "Geyser": 0},
    "Monsoon": {"Lights": 7, "Fans": 6, "AC": 4, "TV": 3, "Geyser": 0},
    "Winter": {"Lights": 7, "Fans": 3, "AC": 0, "TV": 3, "Geyser": 2}
}

SEASON_MONTHS = {
    "Summer": [3, 4, 5, 6], "Monsoon": [7, 8, 9, 10], "Winter": [11, 12, 1, 2]
}

OTHER_APPLIANCES = {
    "Mixer Grinder": 375, "Mobile Charger": 10, "Laptop Charger": 65,
    "Iron Box": 750, "Washing Machine": 750, "Water Purifier": 50, "Motors/Pumps": 375
}


def calculate_tneb_bill(units):
    if units <= 100: return 0
    remaining = units - 100
    bill = 0
    if remaining > 0:
        block = min(remaining, 200)
        bill += block * 4.50
        remaining -= block
    if remaining > 0:
        block = min(remaining, 200)
        bill += block * 6.00
        remaining -= block
    if remaining > 0:
        bill += remaining * 8.00
    return bill


# Survey Inputs
st.subheader(" Household Information")
col1, col2 = st.columns(2)

with col1:
    season = st.selectbox("Select Season:", ["Summer", "Monsoon", "Winter"])
    valid_months = SEASON_MONTHS[season]
    month = st.selectbox("Select Month:", valid_months, format_func=lambda x: calendar.month_name[x])

with col2:
    year = st.number_input("Enter Year:", 2020, 2100, 2024)
    weekend_diff = st.radio("Is weekend usage different?", ["No", "Yes"])

num_days = calendar.monthrange(year, month)[1]
active_appliances = SEASON_APPLIANCES[season]

# Appliance Configuration
st.subheader("🔌 Appliance Configuration")
st.write("**Number of each appliance:**")

appliance_count = {}
appliance_cols = st.columns(3)

for i, app in enumerate(active_appliances):
    with appliance_cols[i % 3]:
        appliance_count[app] = st.number_input(f"{app} Count:", 0, 20, 1, key=f"count_{app}")

# Usage Patterns
st.subheader(" Daily Usage Patterns")
usage_weekday = {};
usage_sat = {};
usage_sun = {}

for app in active_appliances:
    if app == "Refrigerator":
        usage_weekday[app] = usage_sat[app] = usage_sun[app] = 24
        st.write(f" **{app}:** 24 hours/day (continuous operation)")
        continue

    default_val = SEASON_DEFAULT_USAGE[season].get(app, 4)

    if weekend_diff == "No":
        col_a, col_b = st.columns([3, 1])
        with col_a:
            use = st.slider(f"{app} usage (hours/day):", 0.0, 24.0, float(default_val), key=f"use_{app}")
        with col_b:
            st.metric("Hours", f"{use}h")
        usage_weekday[app] = usage_sat[app] = usage_sun[app] = use
    else:
        st.write(f"**{app}:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            usage_weekday[app] = st.slider(f"Mon-Fri:", 0.0, 24.0, float(default_val), key=f"weekday_{app}")
        with col2:
            usage_sat[app] = st.slider(f"Saturday:", 0.0, 24.0, float(default_val + 1), key=f"sat_{app}")
        with col3:
            usage_sun[app] = st.slider(f"Sunday:", 0.0, 24.0, float(default_val - 1), key=f"sun_{app}")

# Small Appliances
st.subheader(" Small Appliances & Miscellaneous")
small_mode = st.radio("Small appliance usage:", ["Use Quick Estimate", "Enter Detailed Usage"], horizontal=True)

other_active = {}
default_small_daily_kwh = 0

if small_mode == "Use Quick Estimate":
    estimate_level = st.select_slider("Select usage level:",
                                      options=["Very Light", "Light", "Moderate", "Heavy", "Very Heavy"],
                                      value="Moderate")
    estimate_values = {"Very Light": 0.2, "Light": 0.35, "Moderate": 0.5, "Heavy": 0.7, "Very Heavy": 0.9}
    default_small_daily_kwh = estimate_values[estimate_level]
    st.info(f"**Estimated small appliance usage:** {default_small_daily_kwh} kWh/day")
else:
    st.write("**Enter weekly usage for small appliances:**")
    small_cols = st.columns(2)
    for i, (app, watt) in enumerate(OTHER_APPLIANCES.items()):
        with small_cols[i % 2]:
            hrs_week = st.number_input(f"{app} (hours/week):", 0.0, 168.0, 2.0, key=f"small_{app}")
            if hrs_week > 0:
                other_active[app] = {"power": watt, "weekly_hours": hrs_week}

# Calculate Bill
if st.button(" Calculate My Electricity Bill", type="primary", use_container_width=True):
    with st.spinner("Calculating your electricity consumption..."):
        daily_records = []

        for day in range(1, num_days + 1):
            weekday = calendar.weekday(year, month, day)

            if weekend_diff == "No":
                today_usage = usage_weekday
            else:
                if weekday < 5:
                    today_usage = usage_weekday
                elif weekday == 5:
                    today_usage = usage_sat
                else:
                    today_usage = usage_sun

            total_kwh = 0

            # Main appliances
            for app in active_appliances:
                power_kw = POWER_RATINGS_MAIN[app] / 1000
                total_kwh += appliance_count[app] * power_kw * today_usage[app]

            # Small appliances
            if small_mode == "Use Quick Estimate":
                total_kwh += default_small_daily_kwh
            else:
                for app, data in other_active.items():
                    power_kw = data["power"] / 1000
                    daily_hours = data["weekly_hours"] / 7
                    total_kwh += power_kw * daily_hours

            daily_records.append({
                "Day": day,
                "Weekday": calendar.day_name[weekday],
                "Energy (kWh)": round(total_kwh, 3)
            })

        df = pd.DataFrame(daily_records)
        total_units = df["Energy (kWh)"].sum()
        bill_amount = calculate_tneb_bill(total_units)

        # Store in session state
        st.session_state.user_data = {
            'season': season,
            'month': month,
            'year': year,
            'appliance_count': appliance_count,
            'usage_pattern': usage_weekday,
            'total_units': total_units,
            'current_bill': bill_amount,
            'daily_breakdown': df
        }

        # Display Results
        st.success(" Calculation Complete! Data saved for optimization recommendations.")

        # Key Metrics
        st.subheader(" Bill Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Consumption", f"{total_units:.1f} kWh")
        with col2:
            st.metric("Electricity Bill", f"₹{bill_amount:.2f}")
        with col3:
            st.metric("Average Daily", f"{total_units / num_days:.1f} kWh")
        with col4:
            slab = "0-100" if total_units <= 100 else "101-300" if total_units <= 300 else "300+"
            st.metric("Tariff Slab", slab)

        # Daily Breakdown
        with st.expander(" Daily Consumption Breakdown"):
            st.dataframe(df, use_container_width=True)

            # Simple chart
            st.bar_chart(df.set_index('Day')['Energy (kWh)'])

        # Bill Analysis
        st.subheader(" Bill Analysis")
        if total_units <= 100:
            st.success("**Great!** You're within the free electricity limit (0-100 units)")
        elif total_units <= 300:
            st.info(f"**Good!** You're in the affordable slab (101-300 units @ ₹4.50/unit)")
        else:
            st.warning(f"**Consider optimizing** - You're in higher tariff slabs (₹6.00-8.00/unit)")

# Alternative navigation section that's always visible
st.markdown("---")
st.subheader(" App Navigation")

col_nav_single = st.columns([1, 2, 1])  # Center the button

with col_nav_single[1]:
    if 'user_data' in st.session_state:
        if st.button("💰 Go to Optimization", use_container_width=True, key="nav_optimization"):
            st.switch_page("pages/optimization.py")
    else:
        st.button("💰 Complete Survey First", use_container_width=True, disabled=True)
        st.caption("Calculate your bill above to enable optimization recommendations")

# Footer with correct page info
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p> Survey Calculator • Use this data for optimization recommendations</p>
    </div>
    """,
    unsafe_allow_html=True
)