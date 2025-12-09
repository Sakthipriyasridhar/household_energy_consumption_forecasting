import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Bill Optimization",
    page_icon="💰",
    layout="wide"
)

# Navigation
col_nav, col_title = st.columns([1, 5])
with col_nav:
    if st.button("← Back to Main"):
        st.switch_page("main.py")
with col_title:
    st.title("💰 Bill Optimization & Savings Analysis")

# ENHANCED APPLIANCE DATABASE
APPLIANCE_DATABASE = {
    "AC": {
        "power_rating": 1500,
        "typical_usage": 8,
        "efficiency_tips": [
            "Set temperature to 24°C instead of 18°C (saves 6-8% per degree)",
            "Use smart thermostat with occupancy sensors",
            "Clean AC filters monthly for 5-10% efficiency gain",
            "Use ceiling fans with AC to raise thermostat by 2-3°C",
            "Close curtains during peak sunlight hours"
        ],
        "upgrade_options": [
            "5-star inverter AC (40% savings) - ₹35,000-50,000",
            "Solar-powered AC system - ₹80,000-1,20,000"
        ],
        "maintenance_tips": [
            "Annual professional servicing - ₹1,500/year",
            "Clean filters every 2 weeks",
            "Check refrigerant levels annually"
        ]
    },
    "Geyser": {
        "power_rating": 2000,
        "typical_usage": 2,
        "efficiency_tips": [
            "Use timer for 1 hour instead of continuous use",
            "Insulate water pipes and tank",
            "Lower thermostat to 50°C instead of 60°C",
            "Take shorter showers (5 minutes max)",
            "Fix leaking taps immediately"
        ],
        "upgrade_options": [
            "Solar water heater (80% savings) - ₹25,000-40,000",
            "Heat pump water heater (60% savings) - ₹30,000-50,000"
        ],
        "maintenance_tips": [
            "Descale every 6 months - ₹500/service",
            "Check anode rod annually",
            "Insulate exposed pipes"
        ]
    },
    "Refrigerator": {
        "power_rating": 150,
        "typical_usage": 24,
        "efficiency_tips": [
            "Ensure proper door seals - test with paper slip method",
            "Maintain 4-5cm clearance from wall for ventilation",
            "Set temperature to 4°C (fridge) and -18°C (freezer)",
            "Defrost regularly if not frost-free",
            "Allow hot food to cool before refrigerating"
        ],
        "upgrade_options": [
            "5-star rated refrigerator (30% savings) - ₹25,000-40,000",
            "Inverter technology models - ₹30,000-50,000"
        ],
        "maintenance_tips": [
            "Clean condenser coils every 3 months",
            "Check door seals every month",
            "Defrost when ice buildup exceeds 1cm"
        ]
    },
    "Washing Machine": {
        "power_rating": 500,
        "typical_usage": 1,
        "efficiency_tips": [
            "Use cold water wash cycles (90% energy saving)",
            "Always run full loads",
            "Use high spin speed to reduce drying time",
            "Clean lint filter after every use",
            "Use eco-mode for lightly soiled clothes"
        ],
        "upgrade_options": [
            "Front-loading machine (50% savings) - ₹20,000-35,000",
            "Inverter technology models - ₹25,000-40,000"
        ],
        "maintenance_tips": [
            "Monthly drum cleaning with hot water",
            "Check hoses for leaks every 6 months",
            "Clean detergent dispenser weekly"
        ]
    },
    "LED Lighting": {
        "power_rating": 10,
        "typical_usage": 6,
        "efficiency_tips": [
            "Replace all incandescent bulbs with LEDs",
            "Use motion sensors in less-used areas",
            "Install dimmers and smart lighting controls",
            "Utilize natural daylight during daytime",
            "Use task lighting instead of room lighting"
        ],
        "upgrade_options": [
            "Smart LED bulbs with automation - ₹500-1,500 per bulb",
            "Solar-powered outdoor lighting - ₹2,000-5,000"
        ],
        "maintenance_tips": [
            "Dust bulbs regularly for maximum brightness",
            "Check for flickering indicating replacement needed"
        ]
    }
}

# Check if survey data exists
if 'user_data' not in st.session_state:
    st.error(" Please complete the Energy Survey first!")
    st.info("The optimization recommendations require your appliance usage data from the survey.")
    if st.button("Take Survey Now"):
        st.switch_page("pages/1_Survey_Calculator.py")
    st.stop()

user_data = st.session_state.user_data

st.success(f"  Analyzing optimization opportunities for {user_data['total_units']:.1f} units monthly consumption")

# Current Bill Display
st.subheader(" Your Current Energy Profile")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Monthly Consumption", f"{user_data['total_units']:.0f} kWh")
with col2:
    st.metric("Current Bill", f"₹{user_data['current_bill']:.0f}")
with col3:
    efficiency = "High" if user_data['total_units'] < 200 else "Medium" if user_data['total_units'] < 350 else "Low"
    st.metric("Efficiency", efficiency)
with col4:
    season_emoji = "☀️" if user_data['season'] == "Summer" else "🌧️" if user_data['season'] == "Monsoon" else "⛄️"
    st.metric("Season", f"{season_emoji} {user_data['season']}")

# ENHANCED SOLAR ANALYSIS
st.subheader(" Solar Rooftop Potential Analysis")


def calculate_solar_potential(roof_area, location="Urban (Chennai/Madurai)"):
    PANEL_AREA = 1.6
    PANEL_POWER = 330
    LOCATION_FACTORS = {
        "Urban (Chennai/Madurai)": 5.2, "Semi-Urban": 5.0, "Rural": 4.8, "Hilly Area": 4.5
    }

    max_panels = int(roof_area / PANEL_AREA)
    usable_panels = min(max_panels, 20)
    system_capacity_kw = (usable_panels * PANEL_POWER) / 1000
    daily_generation = system_capacity_kw * LOCATION_FACTORS[location] * 0.77
    monthly_generation = daily_generation * 30

    return {
        'roof_area': roof_area,
        'recommended_panels': usable_panels,
        'system_capacity_kw': round(system_capacity_kw, 2),
        'monthly_generation_kwh': round(monthly_generation, 2),
    }


def calculate_solar_savings(monthly_consumption, solar_potential, current_bill):
    monthly_generation = solar_potential['monthly_generation_kwh']
    self_consumption = min(monthly_consumption, monthly_generation)
    export_to_grid = max(0, monthly_generation - monthly_consumption)

    remaining_units = max(0, monthly_consumption - self_consumption)
    bill_after_solar = 0
    if remaining_units > 100:
        remaining_after_free = remaining_units - 100
        if remaining_after_free <= 200:
            bill_after_solar = remaining_after_free * 4.5
        elif remaining_after_free <= 400:
            bill_after_solar = 200 * 4.5 + (remaining_after_free - 200) * 6.0
        else:
            bill_after_solar = 200 * 4.5 + 200 * 6.0 + (remaining_after_free - 400) * 8.0

    monthly_savings = current_bill - bill_after_solar
    export_income = export_to_grid * 3.0
    total_monthly_savings = monthly_savings + export_income

    system_cost = solar_potential['system_capacity_kw'] * 55000
    subsidy = system_cost * 0.40
    net_investment = system_cost - subsidy
    payback_years = net_investment / (total_monthly_savings * 12) if total_monthly_savings > 0 else 999

    return {
        'bill_after_solar': bill_after_solar,
        'monthly_savings': monthly_savings,
        'export_income': export_income,
        'total_monthly_savings': total_monthly_savings,
        'system_cost': system_cost,
        'subsidy': subsidy,
        'net_investment': net_investment,
        'payback_years': payback_years
    }


col_solar1, col_solar2 = st.columns(2)

with col_solar1:
    roof_area = st.slider("Available Roof Area (sq meters):", 10.0, 200.0, 30.0, 5.0)

with col_solar2:
    location_type = st.selectbox("Your Location:",
                                 ["Urban (Chennai/Madurai)", "Semi-Urban", "Rural", "Hilly Area"])

if st.button("Calculate Solar Savings", key="solar_calc"):
    with st.spinner("Analyzing solar potential..."):
        solar_potential = calculate_solar_potential(roof_area, location_type)
        solar_savings = calculate_solar_savings(user_data['total_units'], solar_potential, user_data['current_bill'])

        st.session_state.solar_analysis = {
            'potential': solar_potential,
            'savings': solar_savings
        }

# Display Solar Results
if 'solar_analysis' in st.session_state:
    solar_potential = st.session_state.solar_analysis['potential']
    solar_savings = st.session_state.solar_analysis['savings']

    st.success(" Solar Analysis Complete!")

    # Bill Comparison
    st.subheader("💰 Bill Comparison: Current vs With Solar")

    comp_col1, comp_col2, comp_col3 = st.columns(3)

    with comp_col1:
        st.metric("Current Monthly Bill", f"₹{user_data['current_bill']:.0f}")

    with comp_col2:
        st.metric("With Solar Bill", f"₹{solar_savings['bill_after_solar']:.0f}")

    with comp_col3:
        savings_percent = ((user_data['current_bill'] - solar_savings['bill_after_solar']) / user_data[
            'current_bill']) * 100
        st.metric("Monthly Savings", f"₹{solar_savings['total_monthly_savings']:.0f}",
                  f"{savings_percent:.1f}% reduction")

# ENHANCED APPLIANCE-SPECIFIC RECOMMENDATIONS
st.subheader("🔌 Appliance-Specific Optimization")


def calculate_appliance_savings(appliance, count, usage_hours, current_units):
    """Calculate detailed savings for each appliance"""
    if appliance not in APPLIANCE_DATABASE:
        return None

    data = APPLIANCE_DATABASE[appliance]
    current_consumption = (data['power_rating'] * usage_hours * count * 30) / 1000  # kWh/month

    # Calculate optimized consumption (25% reduction typical)
    optimized_consumption = current_consumption * 0.75
    savings_units = current_consumption - optimized_consumption
    savings_rupees = calculate_tneb_bill(current_units + savings_units) - calculate_tneb_bill(current_units)

    return {
        'appliance': appliance,
        'current_consumption': current_consumption,
        'optimized_consumption': optimized_consumption,
        'savings_units': savings_units,
        'savings_rupees': abs(savings_rupees),
        'efficiency_tips': data['efficiency_tips'],
        'upgrade_options': data['upgrade_options'],
        'maintenance_tips': data['maintenance_tips'],
        'priority': 'HIGH' if savings_rupees > 500 else 'MEDIUM' if savings_rupees > 200 else 'LOW'
    }


def calculate_tneb_bill(units):
    """Calculate TNEB bill based on slab rates"""
    try:
        u = float(units)
    except:
        return 0.0
    if u <= 100:
        return 0.0
    remaining = u - 100.0
    bill = 0.0
    # Block 1: 101-300 units
    block = min(remaining, 200.0)
    bill += block * 4.50
    remaining -= block
    # Block 2: 301-500 units
    if remaining > 0:
        block = min(remaining, 200.0)
        bill += block * 6.00
        remaining -= block
    # Block 3: > 500 units
    if remaining > 0:
        bill += remaining * 8.00
    return round(bill, 2)


def get_detailed_appliance_recommendations(user_data):
    """Generate comprehensive appliance-specific recommendations"""
    recommendations = []
    appliance_count = user_data.get('appliance_count', {})
    usage_pattern = user_data.get('usage_pattern', {})
    current_units = user_data.get('total_units', 0)

    # Analyze each appliance
    for appliance in appliance_count:
        if appliance_count[appliance] > 0:
            usage_hours = usage_pattern.get(appliance, APPLIANCE_DATABASE.get(appliance, {}).get('typical_usage', 4))
            savings_data = calculate_appliance_savings(appliance, appliance_count[appliance], usage_hours,
                                                       current_units)
            if savings_data:
                recommendations.append(savings_data)

    # Sort by savings potential
    recommendations.sort(key=lambda x: x['savings_rupees'], reverse=True)
    return recommendations


# Generate and display recommendations
if st.button(" Generate Appliance Optimization Plan", type="primary"):
    with st.spinner("Analyzing your appliances for optimization opportunities..."):
        recommendations = get_detailed_appliance_recommendations(user_data)
        st.session_state.appliance_recommendations = recommendations

if 'appliance_recommendations' in st.session_state:
    recommendations = st.session_state.appliance_recommendations

    total_appliance_savings = sum(rec['savings_rupees'] for rec in recommendations)

    st.success(f" Total Appliance Savings Potential: ₹{total_appliance_savings:.0f}/month")

    # Display detailed recommendations
    for rec in recommendations:
        with st.expander(
                f"🔧 {rec['appliance']} - {rec['priority']} Priority - Savings: ₹{rec['savings_rupees']:.0f}/month",
                expanded=True):

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("###  Consumption Analysis")
                # Consumption comparison chart
                fig, ax = plt.subplots(figsize=(8, 4))
                categories = ['Current', 'After Optimization']
                values = [rec['current_consumption'], rec['optimized_consumption']]
                colors = ['#ff6b6b', '#51cf66']

                bars = ax.bar(categories, values, color=colors, alpha=0.8)
                ax.set_ylabel('Monthly Consumption (kWh)')
                ax.set_title(f'{rec["appliance"]} - Usage Optimization')

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                            f'{value:.1f} kWh', ha='center', va='bottom')

                st.pyplot(fig)

                st.metric("Monthly Savings", f"₹{rec['savings_rupees']:.0f}")

            with col2:
                st.markdown("###  Efficiency Tips")
                for i, tip in enumerate(rec['efficiency_tips'][:3], 1):
                    st.write(f"{i}. {tip}")

                st.markdown("###  Maintenance Tips")
                for tip in rec['maintenance_tips'][:2]:
                    st.write(f"• {tip}")

            st.markdown("###  Upgrade Options")
            upgrade_cols = st.columns(2)
            for i, upgrade in enumerate(rec['upgrade_options']):
                with upgrade_cols[i % 2]:
                    st.info(f"**Option {i + 1}:** {upgrade}")

# QUICK WINS SECTION
st.subheader("⚡ Quick Energy Saving Wins")

quick_wins_cols = st.columns(2)

with quick_wins_cols[0]:
    st.markdown("###  Immediate Actions")
    immediate_actions = [
        {"action": "Switch off appliances at plug point", "savings": "₹100-200/month", "effort": "Low"},
        {"action": "Use natural light during day", "savings": "₹80-150/month", "effort": "Low"},
        {"action": "Optimize refrigerator temperature", "savings": "₹100-180/month", "effort": "Low"},
        {"action": "Fix leaking taps", "savings": "₹50-100/month", "effort": "Medium"},
    ]

    for action in immediate_actions:
        with st.container(border=True):
            st.write(f"**{action['action']}**")
            col_act1, col_act2 = st.columns(2)
            with col_act1:
                st.write(f"Savings: {action['savings']}")
            with col_act2:
                st.write(f"Effort: {action['effort']}")

with quick_wins_cols[1]:
    st.markdown("###  Behavioral Changes")
    behavioral_changes = [
        {"action": "Use ceiling fans instead of AC when possible", "savings": "₹500-800/month", "impact": "High"},
        {"action": "Run full loads in washing machine", "savings": "₹150-250/month", "impact": "Medium"},
        {"action": "Take 5-minute showers", "savings": "₹200-350/month", "impact": "High"},
        {"action": "Use microwave instead of stove for small meals", "savings": "₹80-120/month", "impact": "Low"},
    ]

    for change in behavioral_changes:
        with st.container(border=True):
            st.write(f"**{change['action']}**")
            col_ch1, col_ch2 = st.columns(2)
            with col_ch1:
                st.write(f"Savings: {change['savings']}")
            with col_ch2:
                st.write(f"Impact: {change['impact']}")

# TOTAL SAVINGS SUMMARY
st.subheader(" Total Savings Summary")

if 'solar_analysis' in st.session_state and 'appliance_recommendations' in st.session_state:
    solar_savings = st.session_state.solar_analysis['savings']['total_monthly_savings']
    appliance_savings = sum(rec['savings_rupees'] for rec in st.session_state.appliance_recommendations)
    quick_win_savings = 400  # Estimated from quick wins

    total_savings = solar_savings + appliance_savings + quick_win_savings
    new_bill = max(0, user_data['current_bill'] - total_savings)
    savings_percent = (total_savings / user_data['current_bill']) * 100

    savings_col1, savings_col2, savings_col3, savings_col4 = st.columns(4)

    with savings_col1:
        st.metric("From Appliances", f"₹{appliance_savings:.0f}/month")
    with savings_col2:
        st.metric("From Solar", f"₹{solar_savings:.0f}/month")
    with savings_col3:
        st.metric("From Quick Wins", f"₹{quick_win_savings}/month")
    with savings_col4:
        st.metric("Potential New Bill", f"₹{new_bill:.0f}", f"{savings_percent:.1f}% savings")

    # Savings visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Appliance Optimization', 'Solar Power', 'Quick Wins']
    savings_values = [appliance_savings, solar_savings, quick_win_savings]
    colors = ['#ffa726', '#42a5f5', '#66bb6a']

    bars = ax.bar(categories, savings_values, color=colors, alpha=0.8)
    ax.set_ylabel('Monthly Savings (₹)')
    ax.set_title('Breakdown of Potential Monthly Savings')

    # Add value labels on bars
    for bar, value in zip(bars, savings_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 10,
                f'₹{value:.0f}', ha='center', va='bottom')

    st.pyplot(fig)

# ACTION PLAN
st.subheader(" Your Personalized Action Plan")

if st.button(" Generate Printable Action Plan"):
    with st.spinner("Creating your personalized action plan..."):
        st.success(" Action Plan Generated!")

        action_plan = """
        ##  Personalized Energy Optimization Plan

        ### Immediate Actions (This Week)
        1. Switch off appliances at plug points when not in use
        2. Set AC temperature to 24°C
        3. Use natural light during daytime

        ### Short-term Goals (1 Month)
        1. Replace all bulbs with LED lights
        2. Install timers for water heater
        3. Service AC and refrigerator

        ### Long-term Investments (3-6 Months)
        1. Explore solar rooftop installation
        2. Consider appliance upgrades to 5-star rated models
        3. Install smart home energy monitoring

        **Expected Monthly Savings: ₹{}/month**
        **Expected Annual Savings: ₹{}/year**
        """.format(
            int(total_savings) if 'total_savings' in locals() else "500-1000",
            int(total_savings * 12) if 'total_savings' in locals() else "6000-12000"
        )

        st.download_button(
            label=" Download Action Plan",
            data=action_plan,
            file_name="energy_optimization_plan.txt",
            mime="text/plain"
        )

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p> Start with quick wins today and plan your long-term energy savings strategy</p>
    </div>
    """,
    unsafe_allow_html=True
)