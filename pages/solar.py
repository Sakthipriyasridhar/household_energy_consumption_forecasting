import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Solar Analysis", page_icon="☀️", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .solar-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-highlight {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #00b4d8;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("☀️ Solar Energy Analysis")
st.caption("Analyze your solar potential, savings, and return on investment")

# Get user data
user_data = st.session_state.get("user_data", {})
monthly_consumption = user_data.get("monthly_consumption", 300)
monthly_cost = user_data.get("monthly_cost", 2500)
location = user_data.get("household", {}).get("location", "Chennai")
house_area = user_data.get("household", {}).get("area", 1200)

# Solar Calculator
st.subheader("🏠 Your Solar Potential")

col1, col2, col3 = st.columns(3)

with col1:
    roof_area = st.number_input(
        "Available Roof Area (sq.ft)", 
        min_value=100.0,
        max_value=10000.0,
        value=1000.0,
        step=100.0,
        help="Approximately 30% of total roof area is usable"
    )

with col2:
    solar_budget = st.select_slider(
        "Budget Range",
        options=["₹1-2 Lakhs", "₹2-3 Lakhs", "₹3-5 Lakhs", "₹5-10 Lakhs"],
        value="₹3-5 Lakhs"
    )

with col3:
    shading_factor = st.slider(
        "Roof Shading (%)",
        min_value=0,
        max_value=100,
        value=10,
        help="Percentage of roof affected by shadows"
    )

# Location-specific factors
solar_factors = {
    "Chennai": {"insolation": 5.4, "efficiency": 0.85, "color": "#FF6B6B"},
    "Coimbatore": {"insolation": 5.2, "efficiency": 0.82, "color": "#4ECDC4"},
    "Madurai": {"insolation": 5.5, "efficiency": 0.87, "color": "#FFD166"},
    "Trichy": {"insolation": 5.3, "efficiency": 0.84, "color": "#06D6A0"},
    "Salem": {"insolation": 5.1, "efficiency": 0.81, "color": "#118AB2"},
    "Other": {"insolation": 5.0, "efficiency": 0.80, "color": "#073B4C"}
}

location_factor = solar_factors.get(location, solar_factors["Other"])

# Calculate button
if st.button("🔍 Calculate Solar Potential", type="primary", use_container_width=True):
    # Adjust efficiency based on shading
    shading_effect = 1 - (shading_factor / 100 * 0.5)  # 50% impact of shading
    adjusted_efficiency = location_factor["efficiency"] * shading_effect
    
    # Calculations
    usable_area_m2 = roof_area * 0.0929 * 0.7  # Convert to m², 70% usable
    system_size_kw = usable_area_m2 * 0.15  # 150W per m²
    
    # Daily generation (kWh)
    daily_generation = system_size_kw * location_factor["insolation"] * adjusted_efficiency
    
    # Monthly and annual
    monthly_generation = daily_generation * 30
    annual_generation = monthly_generation * 12
    
    # Consumption coverage
    coverage_percentage = min(100, (monthly_generation / monthly_consumption) * 100) if monthly_consumption > 0 else 0
    
    # Savings (assuming self-consumption priority)
    self_consumption = min(monthly_consumption, monthly_generation * 0.8)
    export = max(0, monthly_generation - self_consumption)
    
    # TNEB export rate ~₹3/kWh, import rate ~₹8/kWh
    monthly_savings = (self_consumption * 8) + (export * 3)
    annual_savings = monthly_savings * 12
    
    # Investment cost (~₹50,000 per kW)
    investment = system_size_kw * 50000
    
    # Subsidy (40% for <3kW, 20% for 3-10kW)
    subsidy_rate = 0.4 if system_size_kw <= 3 else 0.2
    subsidy = investment * subsidy_rate
    net_investment = investment - subsidy
    
    # Payback period
    payback_years = net_investment / annual_savings if annual_savings > 0 else 0
    
    # ROI calculation
    roi_percentage = (annual_savings * 100) / net_investment if net_investment > 0 else 0
    
    # 25-year savings projection
    savings_25yr = 0
    degradation_rate = 0.015  # 1.5% per year
    for year in range(25):
        yearly_saving = annual_savings * ((1 - degradation_rate) ** year)
        savings_25yr += yearly_saving
    
    # Store results
    st.session_state.solar_results = {
        "system_size": system_size_kw,
        "daily_generation": daily_generation,
        "monthly_generation": monthly_generation,
        "annual_generation": annual_generation,
        "coverage_percentage": coverage_percentage,
        "monthly_savings": monthly_savings,
        "annual_savings": annual_savings,
        "investment": investment,
        "subsidy": subsidy,
        "net_investment": net_investment,
        "payback_years": payback_years,
        "roi_percentage": roi_percentage,
        "savings_25yr": savings_25yr,
        "location": location,
        "shading_factor": shading_factor,
        "adjusted_efficiency": adjusted_efficiency
    }

# Display Results with Graphs
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
    
    st.divider()
    
    # ========== GRAPH 1: Monthly Energy Flow ==========
    st.markdown("#### 📈 Monthly Energy Flow Diagram")
    
    # Create energy flow data
    consumption = monthly_consumption
    solar_gen = results['monthly_generation']
    grid_purchase = max(0, consumption - solar_gen * 0.8)
    solar_self_use = min(consumption, solar_gen * 0.8)
    solar_export = max(0, solar_gen - solar_self_use)
    
    fig1 = go.Figure()
    
    # Add bars
    fig1.add_trace(go.Bar(
        name='Your Consumption',
        x=['Energy Sources'],
        y=[consumption],
        marker_color='#2E86AB',
        text=[f'{consumption:.0f} kWh'],
        textposition='auto',
    ))
    
    fig1.add_trace(go.Bar(
        name='Solar Self-Consumption',
        x=['Energy Sources'],
        y=[solar_self_use],
        marker_color='#06D6A0',
        text=[f'{solar_self_use:.0f} kWh'],
        textposition='auto',
    ))
    
    fig1.add_trace(go.Bar(
        name='Grid Purchase',
        x=['Energy Sources'],
        y=[grid_purchase],
        marker_color='#FFD166',
        text=[f'{grid_purchase:.0f} kWh'],
        textposition='auto',
    ))
    
    fig1.update_layout(
        title='Monthly Energy Consumption & Solar Contribution',
        barmode='stack',
        height=400,
        showlegend=True,
        yaxis_title='Energy (kWh)'
    )
    
    col_graph1, col_graph2 = st.columns(2)
    
    with col_graph1:
        st.plotly_chart(fig1, use_container_width=True)
    
    # ========== GRAPH 2: Savings Over Time ==========
    with col_graph2:
        st.markdown("#### 💰 25-Year Savings Projection")
        
        # Calculate cumulative savings over 25 years
        years = list(range(1, 26))
        annual_savings_list = []
        cumulative_savings = 0
        
        for year in years:
            yearly_saving = results['annual_savings'] * ((1 - 0.015) ** (year - 1))
            cumulative_savings += yearly_saving
            annual_savings_list.append(yearly_saving)
        
        cumulative_list = np.cumsum(annual_savings_list)
        
        fig2 = go.Figure()
        
        # Add cumulative savings line
        fig2.add_trace(go.Scatter(
            x=years,
            y=cumulative_list,
            mode='lines+markers',
            name='Cumulative Savings',
            line=dict(color='#06D6A0', width=3),
            fill='tozeroy',
            fillcolor='rgba(6, 214, 160, 0.2)'
        ))
        
        # Add investment line
        fig2.add_trace(go.Scatter(
            x=[0, 25],
            y=[results['net_investment'], results['net_investment']],
            mode='lines',
            name='Net Investment',
            line=dict(color='#EF476F', width=2, dash='dash')
        ))
        
        # Highlight payback year
        payback_year = min(int(results['payback_years']) + 1, 25)
        if payback_year <= 25:
            fig2.add_vline(
                x=payback_year,
                line_dash="dot",
                line_color="#FFD166",
                annotation_text=f"Payback Year {payback_year}",
                annotation_position="top right"
            )
        
        fig2.update_layout(
            title='Cumulative Savings Over 25 Years',
            xaxis_title='Year',
            yaxis_title='Amount (₹)',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    st.divider()
    
    # ========== GRAPH 3: ROI Comparison ==========
    st.markdown("#### 📊 ROI Comparison with Other Investments")
    
    # Create comparison data
    investments = [
        {'Type': 'Solar PV', 'ROI': results['roi_percentage'], 'Risk': 'Low', 'Color': '#06D6A0'},
        {'Type': 'Fixed Deposit', 'ROI': 6.5, 'Risk': 'Very Low', 'Color': '#118AB2'},
        {'Type': 'Mutual Funds', 'ROI': 12.0, 'Risk': 'Moderate', 'Color': '#FFD166'},
        {'Type': 'Stock Market', 'ROI': 15.0, 'Risk': 'High', 'Color': '#EF476F'},
        {'Type': 'Real Estate', 'ROI': 8.0, 'Risk': 'Moderate', 'Color': '#073B4C'}
    ]
    
    df_investments = pd.DataFrame(investments)
    
    fig3 = go.Figure()
    
    fig3.add_trace(go.Bar(
        x=df_investments['Type'],
        y=df_investments['ROI'],
        marker_color=df_investments['Color'],
        text=[f'{roi:.1f}%' for roi in df_investments['ROI']],
        textposition='auto',
    ))
    
    # Highlight solar ROI
    fig3.add_hline(
        y=results['roi_percentage'],
        line_dash="dot",
        line_color="#FFD166",
        annotation_text=f"Your Solar ROI: {results['roi_percentage']:.1f}%",
        annotation_position="top right"
    )
    
    fig3.update_layout(
        title='Return on Investment Comparison',
        xaxis_title='Investment Type',
        yaxis_title='Annual ROI (%)',
        height=400
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    st.divider()
    
    # ========== GRAPH 4: Monthly Generation by Season ==========
    st.markdown("#### 🌦️ Monthly Solar Generation Pattern")
    
    # Create monthly generation pattern based on location
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Seasonal factors for different locations
    seasonal_patterns = {
        'Chennai': [0.95, 0.98, 1.05, 1.10, 1.15, 1.12, 1.05, 1.02, 1.00, 0.98, 0.95, 0.92],
        'Coimbatore': [0.98, 1.00, 1.05, 1.08, 1.10, 1.05, 1.02, 1.00, 0.98, 0.95, 0.92, 0.90],
        'Madurai': [0.95, 0.98, 1.05, 1.12, 1.18, 1.15, 1.08, 1.05, 1.02, 0.98, 0.95, 0.92],
        'Trichy': [0.96, 0.99, 1.06, 1.11, 1.16, 1.13, 1.06, 1.03, 1.01, 0.98, 0.95, 0.93],
        'Salem': [0.97, 1.00, 1.04, 1.09, 1.13, 1.10, 1.04, 1.01, 0.99, 0.96, 0.93, 0.91],
        'Other': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    }
    
    pattern = seasonal_patterns.get(location, seasonal_patterns['Other'])
    monthly_gen_pattern = [results['monthly_generation'] * factor for factor in pattern]
    
    # Temperature data (for secondary axis)
    temp_pattern = {
        'Chennai': [28, 29, 31, 33, 35, 35, 34, 33, 32, 31, 29, 28],
        'Coimbatore': [27, 28, 30, 31, 31, 29, 28, 28, 28, 27, 26, 26],
        'Madurai': [29, 30, 32, 34, 35, 34, 33, 32, 31, 30, 29, 28],
        'Other': [28, 29, 31, 33, 34, 33, 32, 31, 30, 29, 28, 27]
    }
    temps = temp_pattern.get(location, temp_pattern['Other'])
    
    fig4 = go.Figure()
    
    # Add generation bars
    fig4.add_trace(go.Bar(
        x=months,
        y=monthly_gen_pattern,
        name='Solar Generation',
        marker_color='#FFD166',
        text=[f'{val:.0f}' for val in monthly_gen_pattern],
        textposition='auto',
    ))
    
    # Add temperature line
    fig4.add_trace(go.Scatter(
        x=months,
        y=temps,
        name='Avg Temperature (°C)',
        yaxis='y2',
        line=dict(color='#EF476F', width=3),
        mode='lines+markers'
    ))
    
    fig4.update_layout(
        title=f'Monthly Solar Generation Pattern for {location}',
        xaxis_title='Month',
        yaxis_title='Generation (kWh)',
        yaxis2=dict(
            title='Temperature (°C)',
            overlaying='y',
            side='right',
            titlefont=dict(color='#EF476F'),
            tickfont=dict(color='#EF476F')
        ),
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    st.divider()
    
    # ========== GRAPH 5: System Cost Breakdown ==========
    st.markdown("#### 💵 Solar System Cost Breakdown")
    
    # Cost components
    cost_components = [
        {'Component': 'Solar Panels', 'Percentage': 50, 'Color': '#06D6A0'},
        {'Component': 'Inverter', 'Percentage': 20, 'Color': '#118AB2'},
        {'Component': 'Mounting Structure', 'Percentage': 10, 'Color': '#FFD166'},
        {'Component': 'Installation', 'Percentage': 10, 'Color': '#EF476F'},
        {'Component': 'Wiring & Protection', 'Percentage': 5, 'Color': '#073B4C'},
        {'Component': 'Maintenance Fund', 'Percentage': 5, 'Color': '#2E86AB'}
    ]
    
    df_costs = pd.DataFrame(cost_components)
    df_costs['Amount'] = df_costs['Percentage'] * results['investment'] / 100
    
    fig5 = go.Figure(data=[go.Pie(
        labels=df_costs['Component'],
        values=df_costs['Amount'],
        hole=.4,
        marker=dict(colors=df_costs['Color']),
        textinfo='label+percent',
        texttemplate='%{label}<br>₹%{value:,.0f}<br>(%{percent})',
        hovertemplate='<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent} of total'
    )])
    
    fig5.update_layout(
        title='Investment Cost Breakdown',
        height=400,
        showlegend=False,
        annotations=[dict(
            text=f"Total:<br>₹{results['investment']:,.0f}",
            x=0.5, y=0.5, font_size=16, showarrow=False
        )]
    )
    
    col_graph5, col_graph6 = st.columns(2)
    
    with col_graph5:
        st.plotly_chart(fig5, use_container_width=True)
    
    # ========== GRAPH 6: Environmental Impact ==========
    with col_graph6:
        st.markdown("#### 🌍 Environmental Impact")
        
        # Environmental metrics
        co2_saved_monthly = results['monthly_generation'] * 0.85  # kg CO2 per kWh
        co2_saved_annual = co2_saved_monthly * 12
        co2_saved_25yr = co2_saved_annual * 25
        
        trees_equivalent = co2_saved_annual / 21  # Average tree absorbs 21kg CO2/year
        cars_equivalent = co2_saved_annual / 4600  # Average car emits 4.6 tons/year
        
        environmental_data = {
            'Metric': ['CO₂ Saved (Annual)', 'Equivalent Trees', 'Cars Off Road'],
            'Value': [co2_saved_annual, trees_equivalent, cars_equivalent],
            'Unit': ['kg CO₂', 'Trees', 'Cars'],
            'Color': ['#06D6A0', '#118AB2', '#FFD166']
        }
        
        df_env = pd.DataFrame(environmental_data)
        
        fig6 = go.Figure()
        
        fig6.add_trace(go.Bar(
            x=df_env['Metric'],
            y=df_env['Value'],
            marker_color=df_env['Color'],
            text=[f'{val:.0f} {unit}' for val, unit in zip(df_env['Value'], df_env['Unit'])],
            textposition='auto',
        ))
        
        fig6.update_layout(
            title='Annual Environmental Benefits',
            height=400,
            yaxis_title='',
            xaxis_title=''
        )
        
        st.plotly_chart(fig6, use_container_width=True)
    
    st.divider()
    
    # ========== TABBED DETAILED ANALYSIS ==========
    st.subheader("📋 Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Financial Analysis", "Technical Details", "Next Steps"])
    
    with tab1:
        st.markdown(f"""
        **Financial Summary:**
        - **Total Investment:** ₹{results['investment']:,.0f}
        - **Government Subsidy:** ₹{results['subsidy']:,.0f} ({'40%' if results['system_size'] <= 3 else '20%'})
        - **Net Investment:** ₹{results['net_investment']:,.0f}
        - **Annual Savings:** ₹{results['annual_savings']:,.0f}
        - **Annual ROI:** {results['roi_percentage']:.1f}%
        - **25-Year Total Savings:** ₹{results['savings_25yr']:,.0f}
        
        **Key Insight:**
        Your solar system will pay for itself in **{results['payback_years']:.1f} years** and generate 
        **₹{results['savings_25yr'] - results['net_investment']:,.0f} in pure profit** over 25 years!
        """)
    
    with tab2:
        st.markdown(f"""
        **Technical Specifications:**
        - **System Size:** {results['system_size']:.1f} kW
        - **Number of Panels:** Approximately {int(results['system_size'] * 1000 / 330)} (330W panels)
        - **Roof Area Required:** {roof_area:.0f} sq.ft
        - **Daily Generation:** {results['daily_generation']:.1f} kWh
        - **Annual Generation:** {results['annual_generation']:.0f} kWh
        - **Consumption Coverage:** {results['coverage_percentage']:.0f}%
        - **Location Factor:** {location_factor['insolation']} kWh/m²/day
        - **System Efficiency:** {results['adjusted_efficiency']*100:.1f}%
        
        **System Components:**
        - Solar Panels: Monocrystalline, 330W each
        - Inverter: Grid-tied with MPPT technology
        - Mounting: Galvanized steel structure
        - Monitoring: Cloud-based performance tracking
        """)
    
    with tab3:
        steps = [
            "1. **Site Survey:** Get professional assessment (free)",
            "2. **Quotation:** Compare quotes from 3+ TNERC-approved installers",
            "3. **Net Metering Application:** Submit to TNEB (takes 15-20 days)",
            "4. **Installation:** 3-5 working days",
            "5. **Commissioning:** System testing and TNEB inspection",
            "6. **Subsidy Claim:** Submit documents to MNRE portal"
        ]
        
        for step in steps:
            st.markdown(step)
        
        st.markdown("""
        **Recommended Installers in Tamil Nadu:**
        - Tata Power Solar
        - Waaree Energies
        - Vikram Solar
        - Loom Solar
        - Local TNERC-approved installers
        """)
    
    # Action Buttons
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate Detailed Report", use_container_width=True, icon="📊"):
            st.success("Detailed report generated! (PDF download would be implemented here)")
    
    with col2:
        if st.button("📱 Share Analysis", use_container_width=True, icon="📤"):
            st.success("Analysis link copied to clipboard!")
    
    with col3:
        if st.button("🔄 Recalculate", use_container_width=True):
            del st.session_state.solar_results
            st.rerun()

else:
    # Show calculator with sample visualization
    st.info("👆 Enter your details and click 'Calculate Solar Potential' to see detailed analysis")
    
    col_sample1, col_sample2 = st.columns(2)
    
    with col_sample1:
        st.markdown("### 📊 What to Expect")
        st.markdown("""
        **Visual Analysis Includes:**
        
        1. **Energy Flow Diagram** - How solar meets your needs
        2. **Savings Projection** - 25-year financial outlook
        3. **ROI Comparison** - Solar vs other investments
        4. **Monthly Generation** - Seasonal patterns
        5. **Cost Breakdown** - Where your money goes
        6. **Environmental Impact** - Your carbon footprint reduction
        
        **Sample Results for Chennai:**
        - 3 kW system: ₹1.5L investment
        - Monthly savings: ₹3,200
        - Payback: 3.5 years
        - 25-year profit: ₹7.5L
        """)
    
    with col_sample2:
        # Sample mini chart
        fig_sample = go.Figure()
        
        # Sample savings projection
        years = list(range(1, 11))
        sample_savings = [3200 * 12 * ((1 - 0.015) ** (y-1)) for y in years]
        cumulative = np.cumsum(sample_savings)
        
        fig_sample.add_trace(go.Scatter(
            x=years,
            y=cumulative,
            mode='lines+markers',
            name='Cumulative Savings',
            line=dict(color='#06D6A0', width=3),
            fill='tozeroy',
            fillcolor='rgba(6, 214, 160, 0.2)'
        ))
        
        fig_sample.add_hline(
            y=150000,
            line_dash="dash",
            line_color="#EF476F",
            annotation_text="Investment: ₹1.5L",
            annotation_position="top right"
        )
        
        fig_sample.update_layout(
            title='Sample: 10-Year Savings Projection',
            xaxis_title='Year',
            yaxis_title='Cumulative Savings (₹)',
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig_sample, use_container_width=True)

# Footer
st.divider()
st.caption("☀️ *Disclaimer: This analysis provides estimates based on typical conditions. Actual performance may vary based on specific site conditions, equipment quality, and maintenance practices.*")
