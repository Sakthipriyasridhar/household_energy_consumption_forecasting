import streamlit as st
import numpy as np

st.set_page_config(page_title="Solar Analysis", page_icon="☀️")
st.title("☀️ Solar Energy Analysis")

# Get user data
user_data = st.session_state.get("user_data", {})
monthly_consumption = user_data.get("monthly_consumption", 300)
monthly_cost = user_data.get("monthly_cost", 2500)
location = user_data.get("household", {}).get("location", "Chennai")
house_area = user_data.get("household", {}).get("area", 1200)

# Solar Calculator
st.subheader("🏠 Your Solar Potential")

col1, col2 = st.columns(2)

with col1:
    roof_area = st.number_input(
    "Available Roof Area (sq.ft)", 
    min_value=100.0,  # Use float instead of int
    max_value=10000.0,  # Add max_value if needed
    value=1000.0,  # Add a default value
    step=100.0,  # Make step a float
    help="Approximately 30% of total roof area is usable")

with col2:
    solar_budget = st.select_slider("Budget Range",
                                   options=["₹1-2 Lakhs", "₹2-3 Lakhs", "₹3-5 Lakhs", "₹5-10 Lakhs"],
                                   value="₹3-5 Lakhs")

# Location-specific factors
solar_factors = {
    "Chennai": {"insolation": 5.4, "efficiency": 0.85},
    "Coimbatore": {"insolation": 5.2, "efficiency": 0.82},
    "Madurai": {"insolation": 5.5, "efficiency": 0.87},
    "Trichy": {"insolation": 5.3, "efficiency": 0.84},
    "Salem": {"insolation": 5.1, "efficiency": 0.81},
    "Other": {"insolation": 5.0, "efficiency": 0.80}
}

location_factor = solar_factors.get(location, solar_factors["Other"])

# Calculate solar potential
if st.button("🔍 Calculate Solar Potential", type="primary"):
    # Calculations
    usable_area_m2 = roof_area * 0.0929 * 0.7  # Convert to m², 70% usable
    system_size_kw = usable_area_m2 * 0.15  # 150W per m²
    
    # Daily generation (kWh)
    daily_generation = system_size_kw * location_factor["insolation"] * location_factor["efficiency"]
    
    # Monthly and annual
    monthly_generation = daily_generation * 30
    annual_generation = monthly_generation * 12
    
    # Savings (assuming 70% self-consumption, 30% export)
    self_consumption = min(monthly_consumption, monthly_generation * 0.7)
    export = max(0, monthly_generation * 0.3)
    
    # TNEB export rate ~₹3/kWh, import rate ~₹8/kWh
    monthly_savings = (self_consumption * 8) + (export * 3)
    
    # Investment cost (~₹50,000 per kW)
    investment = system_size_kw * 50000
    
    # Subsidy (40% for <3kW, 20% for 3-10kW)
    subsidy_rate = 0.4 if system_size_kw <= 3 else 0.2
    subsidy = investment * subsidy_rate
    net_investment = investment - subsidy
    
    # Payback period
    payback_years = net_investment / (monthly_savings * 12)
    
    # Store results
    st.session_state.solar_results = {
        "system_size": system_size_kw,
        "monthly_generation": monthly_generation,
        "monthly_savings": monthly_savings,
        "investment": investment,
        "subsidy": subsidy,
        "net_investment": net_investment,
        "payback_years": payback_years,
        "roi_percentage": (monthly_savings * 12 * 100) / net_investment if net_investment > 0 else 0
    }

# Display Results
if st.session_state.get("solar_results"):
    results = st.session_state.solar_results
    
    st.subheader("📊 Solar Analysis Results")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Size", f"{results['system_size']:.1f} kW")
    
    with col2:
        st.metric("Monthly Generation", f"{results['monthly_generation']:.0f} kWh")
    
    with col3:
        st.metric("Monthly Savings", f"₹{results['monthly_savings']:,.0f}")
    
    with col4:
        st.metric("Payback Period", f"{results['payback_years']:.1f} years")
    
    # Detailed Breakdown
    st.subheader("💰 Financial Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Cost Breakdown", "Savings Projection", "ROI Analysis"])
    
    with tab1:
        st.markdown(f"""
        **Initial Investment:**
        - Total System Cost: ₹{results['investment']:,.0f}
        - Government Subsidy (40%): ₹{results['subsidy']:,.0f}
        - Net Investment: ₹{results['net_investment']:,.0f}
        
        **Additional Benefits:**
        - 25-year system lifespan
        - 5-year warranty standard
        - Increased property value
        - Carbon offset: {results['monthly_generation'] * 0.85:.0f} kg CO₂/month
        """)
    
    with tab2:
        years = 10
        savings_projection = []
        cumulative_savings = 0
        
        for year in range(1, years + 1):
            yearly_saving = results['monthly_savings'] * 12 * (0.985 ** (year - 1))  # 1.5% degradation
            cumulative_savings += yearly_saving
            savings_projection.append({
                "Year": year,
                "Annual Savings": f"₹{yearly_saving:,.0f}",
                "Cumulative": f"₹{cumulative_savings:,.0f}"
            })
        
        import pandas as pd
        df = pd.DataFrame(savings_projection)
        st.dataframe(df, use_container_width=True)
        
        # Highlight breakeven
        if results['payback_years'] <= years:
            st.success(f"✅ Breakeven achieved in Year {int(results['payback_years']) + 1}")
    
    with tab3:
        roi = results['roi_percentage']
        st.metric("Annual ROI", f"{roi:.1f}%")
        
        if roi > 15:
            st.success("Excellent investment (>15% ROI)")
        elif roi > 10:
            st.info("Good investment (10-15% ROI)")
        else:
            st.warning("Moderate investment (<10% ROI)")
        
        st.markdown(f"""
        **Comparison with Other Investments:**
        - Fixed Deposit: 6-7% ROI
        - Mutual Funds: 10-12% ROI (risky)
        - Solar: {roi:.1f}% ROI (guaranteed, tax-free)
        
        **Additional Benefits:**
        - Inflation protection (electricity prices increase 5-7%/year)
        - Grid independence during outages
        - Environmental contribution
        """)
    
    # Government Schemes
    st.subheader("🏛️ Government Schemes & Subsidies")
    
    schemes = [
        {
            "name": "PM Surya Ghar Yojana",
            "subsidy": "40% for systems up to 3kW, 20% for 3-10kW",
            "eligibility": "All residential consumers",
            "link": "https://pmsuryaghar.gov.in"
        },
        {
            "name": "Tamil Nadu Solar Policy",
            "subsidy": "Additional state incentives",
            "eligibility": "TN residents, own roof required",
            "link": "https://www.teda.in"
        },
        {
            "name": "Net Metering",
            "subsidy": "Feed-in tariff for excess generation",
            "eligibility": "Grid-connected systems",
            "link": "https://www.tnebnet.org"
        }
    ]
    
    for scheme in schemes:
        with st.expander(f"**{scheme['name']}**"):
            st.markdown(f"""
            **Subsidy:** {scheme['subsidy']}
            
            **Eligibility:** {scheme['eligibility']}
            
            **More Info:** [{scheme['link']}]({scheme['link']})
            """)
    
    # Next Steps
    st.subheader("📋 Next Steps")
    
    steps = [
        "1. **Site Survey:** Get professional assessment of your roof",
        "2. **Quotation:** Compare quotes from 3+ installers",
        "3. **Approval:** Apply for net metering with TNEB",
        "4. **Installation:** 3-5 days installation time",
        "5. **Commissioning:** System testing and grid connection",
        "6. **Subsidy Claim:** Submit documents for subsidy"
    ]
    
    for step in steps:
        st.markdown(step)
    
    # Action Buttons
    st.divider()
    col1, col2= st.columns(2)
    

    
    with col1:
        if st.button("📄 Download Report", use_container_width=True):
            st.success("Report generated for download!")
    
    with col2:
        if st.button("🔄 Recalculate", use_container_width=True):
            del st.session_state.solar_results
            st.rerun()

else:
    # Show calculator
    st.info("👆 Enter your details and click 'Calculate Solar Potential'")
    
    with st.expander("📈 Sample Analysis"):
        st.markdown("""
        **For a typical Chennai household:**
        
        - **Roof Area:** 1000 sq.ft (300 sq.ft usable)
        - **System Size:** 3 kW
        - **Monthly Generation:** 450 kWh
        - **Monthly Savings:** ₹3,200
        - **Investment:** ₹1,50,000
        - **Subsidy:** ₹60,000 (40%)
        - **Net Cost:** ₹90,000
        - **Payback:** 2.3 years
        - **25-year Savings:** ₹9,60,000
        """)

# Footer
st.divider()
st.caption("☀️ Note: Actual savings may vary based on installation quality, maintenance, and weather conditions")
