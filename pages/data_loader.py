# pages/2_Data_Loader.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(
    page_title="Load Energy Data",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .option-card {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px solid #e0e0e0;
        transition: all 0.3s;
    }
    .option-card:hover {
        border-color: #00b4d8;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .data-preview {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
    }
    .date-picker-container {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Energy Data Manager")
st.markdown("### Load, Upload, or Generate Energy Consumption Data")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'prediction_settings' not in st.session_state:
    st.session_state.prediction_settings = {
        'prediction_months': 12,
        'start_date': None,
        'data_type': None
    }

# Function to generate survey-based data with user-selected dates
def generate_survey_based_data(start_date=None, months=12):
    """Generate realistic data based on survey responses with dynamic dates"""
    if 'user_data' not in st.session_state or not st.session_state.user_data:
        return None
    
    # Get current date
    today = datetime.now()
    
    # If no start date provided, use 1 year ago from today
    if start_date is None:
        start_date = today - timedelta(days=365)
    
    # Calculate end date based on months
    if isinstance(months, int):
        end_date = start_date + timedelta(days=months*30)
    else:
        end_date = start_date + timedelta(days=365)  # Default 1 year
    
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Get survey data
    user_data = st.session_state.user_data
    
    # Extract key parameters from survey
    monthly_usage = user_data.get('monthly_consumption', 900)  # Default 900 kWh
    if monthly_usage == 0:
        monthly_usage = 900  # Fallback
    
    location = user_data.get('household', {}).get('location', 'Chennai')
    num_appliances = len(user_data.get('appliances', []))
    ac_months = user_data.get('usage', {}).get('ac_months', 6)
    
    # Calculate daily average from monthly
    daily_avg = monthly_usage / 30
    
    # Generate consumption pattern
    consumption = []
    temperatures = []
    
    for i, date in enumerate(dates):
        # Base consumption
        base = daily_avg
        
        # Day of week pattern (higher on weekdays)
        day_of_week = date.weekday()
        if day_of_week < 5:  # Weekday
            base *= 1.15
        else:  # Weekend
            base *= 0.85
        
        # Month of year pattern (seasonality)
        month = date.month
        day_of_year = date.dayofyear
        
        # Enhanced seasonality based on location
        if location.lower() in ['chennai', 'madurai', 'hyderabad']:  # Hot cities
            # Strong summer peak
            if month in [3, 4, 5, 6]:  # Summer
                seasonal_factor = 1.4
            elif month in [7, 8, 9]:  # Monsoon
                seasonal_factor = 1.2
            elif month in [10, 11]:  # Post-monsoon
                seasonal_factor = 1.0
            else:  # Winter
                seasonal_factor = 0.9
        else:  # Other locations
            # Moderate seasonality
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        base *= seasonal_factor
        
        # Appliance factor
        appliance_factor = 1 + (num_appliances * 0.03)
        base *= appliance_factor
        
        # Random variation (10-20% of base)
        random_variation = np.random.normal(0, base * 0.15)
        final_consumption = max(base + random_variation, daily_avg * 0.5)
        
        # Generate realistic temperature based on location and season
        if location.lower() in ['chennai', 'madurai', 'hyderabad']:
            # Hot climate
            base_temp = 28 + 8 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
        elif location.lower() in ['bangalore', 'coimbatore']:
            # Moderate climate
            base_temp = 22 + 6 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
        else:
            # Default
            base_temp = 25 + 7 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
        
        temp = base_temp + np.random.normal(0, 3)
        temp = max(min(temp, 45), 10)  # Clamp between 10-45°C
        
        consumption.append(round(final_consumption, 2))
        temperatures.append(round(temp, 1))
    
    # Create dataframe
    df = pd.DataFrame({
        'Date': dates,
        'Energy_Consumption_kWh': consumption,
        'Temperature_C': temperatures,
        'Source': 'Survey Prediction',
        'Location': location,
        'Household_Size': user_data.get('household', {}).get('members', 4)
    })
    
    # Store prediction settings
    st.session_state.prediction_settings.update({
        'start_date': start_date,
        'prediction_months': months,
        'data_type': 'survey'
    })
    
    return df

# Function to parse uploaded file with date selection
def parse_uploaded_file(uploaded_file, start_date=None, months=12):
    """Parse uploaded file and prepare for analysis"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Try to parse date columns
        date_cols = [col for col in df.columns if any(keyword in col.lower() 
                                                    for keyword in ['date', 'time', 'day', 'timestamp'])]
        
        if date_cols:
            for col in date_cols:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    continue
        
        return df, None
    except Exception as e:
        return None, str(e)

# Function to generate sample data
def generate_sample_data(start_date=None, months=12):
    """Generate realistic sample data with dynamic dates"""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    
    end_date = start_date + timedelta(days=months*30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base pattern
    base_daily = 30  # Average 30 kWh per day
    
    # Complex seasonality
    seasonal = 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    
    # Weekly pattern
    weekly = np.where(dates.weekday < 5, 4, -4)
    
    # Trend (slight increase over time)
    trend = np.linspace(0, 3, len(dates))
    
    # Holidays (random low consumption days)
    holidays = np.zeros(len(dates))
    holiday_indices = np.random.choice(len(dates), size=10, replace=False)
    holidays[holiday_indices] = -10
    
    # Noise
    noise = np.random.normal(0, 2.5, len(dates))
    
    consumption = base_daily + seasonal + weekly + trend + holidays + noise
    consumption = np.maximum(consumption, 15)  # Minimum
    
    # Temperature with seasonality
    temp_base = 20 + 12 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    temp_noise = np.random.normal(0, 4, len(dates))
    temperature = temp_base + temp_noise
    temperature = np.maximum(temperature, 5)
    
    df = pd.DataFrame({
        'Date': dates,
        'Energy_Consumption_kWh': np.round(consumption, 2),
        'Temperature_C': np.round(temperature, 1),
        'Source': 'Sample Data',
        'Location': 'Sample City',
        'Household_Size': 4
    })
    
    st.session_state.prediction_settings.update({
        'start_date': start_date,
        'prediction_months': months,
        'data_type': 'sample'
    })
    
    return df

# Main interface
st.markdown("### 📋 Select Data Source")

# Create tabs for different data sources
tab1, tab2, tab3 = st.tabs(["📊 Use Survey Data", "📁 Upload Your Data", "🧪 Try Sample Data"])

with tab1:
    st.markdown("""
    <div class="option-card">
        <h3>📋 Generate Data from Survey</h3>
        <p>Create realistic energy consumption patterns based on your survey responses</p>
        <p><small>Customize the time period and generate personalized data</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'survey_completed' in st.session_state and st.session_state.survey_completed:
        # Date selection for survey data
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date for Data",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now(),
                key="survey_start"
            )
        
        with col2:
            prediction_months = st.number_input(
                "Months to Generate/Predict",
                min_value=1,
                max_value=36,
                value=12,
                help="Number of months of data to generate"
            )
        
        if st.button("🚀 Generate Survey Data", type="primary", use_container_width=True):
            with st.spinner("Generating personalized data from survey..."):
                df = generate_survey_based_data(
                    start_date=start_date,
                    months=prediction_months
                )
                
                if df is not None:
                    st.session_state.forecast_data = df
                    st.session_state.data_loaded = True
                    st.success(f"✅ Generated {len(df)} days of personalized data!")
                    st.rerun()
                else:
                    st.error("Could not generate data. Please check your survey responses.")
    
    else:
        st.warning("⚠️ No completed survey found.")
        if st.button("Take Survey Now", type="secondary"):
            st.switch_page("pages/survey.py")

with tab2:
    st.markdown("""
    <div class="option-card">
        <h3>📁 Upload Your Historical Data</h3>
        <p>Upload your own energy consumption records for analysis</p>
        <p><small>Supports CSV, Excel (.xlsx, .xls) with date and consumption columns</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload your energy data file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload CSV or Excel file with date and consumption columns"
    )
    
    if uploaded_file is not None:
        st.success(f"📄 File uploaded: {uploaded_file.name}")
        
        # Parse the file
        df, error = parse_uploaded_file(uploaded_file)
        
        if error:
            st.error(f"❌ Error: {error}")
        elif df is not None:
            # Show preview
            st.markdown("#### 📋 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            st.markdown("#### 🎯 Column Mapping")
            
            # Auto-detect columns
            date_columns = [col for col in df.columns if any(keyword in col.lower() 
                                                           for keyword in ['date', 'time', 'day', 'timestamp'])]
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Date column selection
                date_col = st.selectbox(
                    "Date/Time Column",
                    options=['Select column...'] + df.columns.tolist(),
                    index=date_columns[0] + 1 if date_columns else 1,
                    help="Select the column containing dates or timestamps"
                )
            
            with col_b:
                # Consumption column selection
                consumption_col = st.selectbox(
                    "Energy Consumption Column (kWh)",
                    options=['Select column...'] + numeric_columns,
                    index=numeric_columns[0] + 1 if numeric_columns else 1,
                    help="Select the column containing energy consumption values"
                )
            
            # Additional settings for uploaded data
            st.markdown("#### ⚙️ Analysis Settings")
            
            col_c, col_d = st.columns(2)
            
            with col_c:
                future_months = st.number_input(
                    "Months to Predict",
                    min_value=1,
                    max_value=24,
                    value=6,
                    help="How many months into the future to forecast"
                )
            
            with col_d:
                if date_col != 'Select column...':
                    # Show date range of uploaded data
                    try:
                        df_temp = df.copy()
                        df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                        min_date = df_temp[date_col].min()
                        max_date = df_temp[date_col].max()
                        st.info(f"Data range: {min_date.strftime('%b %d, %Y')} to {max_date.strftime('%b %d, %Y')}")
                    except:
                        pass
            
            if st.button("✅ Process and Use This Data", type="primary", use_container_width=True,
                        disabled=(date_col == 'Select column...' or consumption_col == 'Select column...')):
                try:
                    # Clean and prepare data
                    df_clean = df.copy()
                    
                    # Convert date column
                    df_clean['Date'] = pd.to_datetime(df_clean[date_col])
                    
                    # Convert consumption column
                    df_clean['Energy_Consumption_kWh'] = pd.to_numeric(
                        df_clean[consumption_col], 
                        errors='coerce'
                    )
                    
                    # Drop invalid rows
                    df_clean = df_clean.dropna(subset=['Date', 'Energy_Consumption_kWh'])
                    
                    # Sort by date
                    df_clean = df_clean.sort_values('Date').reset_index(drop=True)
                    
                    # Add source info
                    df_clean['Source'] = 'Uploaded Data'
                    
                    # Add location if available
                    if 'Location' not in df_clean.columns and 'location' not in df_clean.columns:
                        df_clean['Location'] = 'Unknown'
                    
                    # Store in session state
                    st.session_state.forecast_data = df_clean
                    st.session_state.data_loaded = True
                    st.session_state.prediction_settings.update({
                        'prediction_months': future_months,
                        'start_date': df_clean['Date'].max(),
                        'data_type': 'uploaded'
                    })
                    
                    st.success(f"✅ Data processed successfully! {len(df_clean)} records loaded")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")

with tab3:
    st.markdown("""
    <div class="option-card">
        <h3>🧪 Generate Sample Data</h3>
        <p>Try the forecasting features with realistic sample data</p>
        <p><small>Perfect for testing and demonstration purposes</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Date selection for sample data
    col1, col2 = st.columns(2)
    
    with col1:
        sample_start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now(),
            key="sample_start"
        )
    
    with col2:
        sample_months = st.number_input(
            "Months of Data",
            min_value=1,
            max_value=36,
            value=12,
            key="sample_months"
        )
    
    if st.button("🧪 Generate Sample Data", type="primary", use_container_width=True):
        with st.spinner("Generating realistic sample data..."):
            df = generate_sample_data(
                start_date=sample_start_date,
                months=sample_months
            )
            
            st.session_state.forecast_data = df
            st.session_state.data_loaded = True
            st.success(f"✅ Generated {len(df)} days of sample data!")
            st.rerun()

# Data Preview Section
st.markdown("---")
st.markdown("### 📈 Current Data Status")

if st.session_state.data_loaded and st.session_state.forecast_data is not None:
    data = st.session_state.forecast_data
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    
    with col2:
        date_range = f"{data['Date'].min().strftime('%b %d, %Y')} to {data['Date'].max().strftime('%b %d, %Y')}"
        st.metric("Date Range", date_range)
    
    with col3:
        avg_consumption = data['Energy_Consumption_kWh'].mean()
        st.metric("Avg Daily", f"{avg_consumption:.1f} kWh")
    
    with col4:
        data_source = st.session_state.prediction_settings.get('data_type', 'Unknown')
        st.metric("Source", data_source.title())
    
    # Data preview in tabs
    tab_preview, tab_stats, tab_viz = st.tabs(["📋 Data Preview", "📊 Statistics", "📈 Visualization"])
    
    with tab_preview:
        st.dataframe(data.head(20), use_container_width=True)
        
        # Show column info
        st.markdown("#### 📝 Column Information")
        col_info = pd.DataFrame({
            'Column': data.columns,
            'Data Type': data.dtypes.astype(str),
            'Non-Null Count': data.notnull().sum().values,
            'Unique Values': [data[col].nunique() for col in data.columns]
        })
        st.dataframe(col_info, use_container_width=True)
    
    with tab_stats:
        # Basic statistics
        st.markdown("#### 📊 Consumption Statistics")
        stats = data['Energy_Consumption_kWh'].describe()
        st.write(stats)
        
        # Monthly statistics
        if len(data) > 30:
            data_monthly = data.copy()
            data_monthly['YearMonth'] = data_monthly['Date'].dt.to_period('M')
            monthly_stats = data_monthly.groupby('YearMonth')['Energy_Consumption_kWh'].agg(['mean', 'min', 'max', 'sum'])
            st.markdown("#### 📅 Monthly Statistics")
            st.dataframe(monthly_stats, use_container_width=True)
    
    with tab_viz:
        # Create time series plot
        fig = go.Figure()
        
        # Add consumption line
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Energy_Consumption_kWh'],
            mode='lines',
            name='Energy Consumption',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='Date: %{x}<br>Consumption: %{y:.1f} kWh<extra></extra>'
        ))
        
        # Add 7-day moving average
        if len(data) > 7:
            data_sorted = data.sort_values('Date')
            moving_avg = data_sorted['Energy_Consumption_kWh'].rolling(window=7).mean()
            fig.add_trace(go.Scatter(
                x=data_sorted['Date'],
                y=moving_avg,
                mode='lines',
                name='7-Day Moving Avg',
                line=dict(color='#ff7f0e', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title='Energy Consumption Over Time',
            xaxis_title='Date',
            yaxis_title='Energy Consumption (kWh)',
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Navigation button
    st.markdown("---")
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    
    with col_nav2:
        if st.button("🚀 Proceed to Forecasting", type="primary", use_container_width=True, key="goto_forecast"):
            # Store prediction settings
            if 'prediction_settings' in st.session_state:
                st.session_state.prediction_months = st.session_state.prediction_settings.get('prediction_months', 12)
            
            st.switch_page("pages/forecast.py")
    
    # Data export option
    st.markdown("### 💾 Export Data")
    csv_data = data.to_csv(index=False)
    st.download_button(
        label="📥 Download Current Data (CSV)",
        data=csv_data,
        file_name=f"energy_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download the current dataset as CSV"
    )
    
else:
    st.info("👈 Select a data source option above to begin")
    
    # Quick tips
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    
    tips_col1, tips_col2, tips_col3 = st.columns(3)
    
    with tips_col1:
        st.markdown("""
        **📋 Survey Data**
        - Based on your household profile
        - Personalized consumption patterns
        - Seasonal variations included
        """)
    
    with tips_col2:
        st.markdown("""
        **📁 Upload Data**
        - Use your historical records
        - More accurate predictions
        - Supports CSV/Excel formats
        """)
    
    with tips_col3:
        st.markdown("""
        **🧪 Sample Data**
        - Perfect for testing
        - Realistic patterns
        - No setup required
        """)

# Footer
st.markdown("---")
st.caption("💡 Tip: For best results, upload at least 1 year of historical data or complete the detailed survey for personalized predictions")
