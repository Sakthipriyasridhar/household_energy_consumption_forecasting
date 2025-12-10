# pages/2_Data_Loader.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
</style>
""", unsafe_allow_html=True)

st.title("📊 Load Energy Data")
st.markdown("### Choose how you want to provide energy consumption data")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None

# Option 1: Use Survey Data (if exists)
def load_survey_data():
    """Load data from survey results"""
    if 'user_data' in st.session_state and st.session_state.user_data:
        # Extract survey data
        user_data = st.session_state.user_data
        
        # Generate sample data based on survey responses
        monthly_usage = user_data.get('monthly_usage', 900)  # Default 900 kWh
        num_appliances = len(user_data.get('appliances', []))
        
        # Create 1 year of synthetic data based on survey
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        
        # Base pattern from survey
        base_daily = monthly_usage / 30  # Average daily usage
        
        # Add seasonality
        seasonal = 0.3 * np.sin(2 * np.pi * np.arange(365) / 365) * base_daily
        
        # Add weekday/weekend pattern
        weekday_pattern = np.where(dates.weekday < 5, 1.1, 0.9)
        
        # Add appliance-specific noise
        appliance_factor = 1 + (num_appliances * 0.05)
        
        # Generate consumption
        consumption = (base_daily + seasonal) * weekday_pattern * appliance_factor
        consumption = np.maximum(consumption, base_daily * 0.7)  # Minimum threshold
        
        # Add random noise
        noise = np.random.normal(0, base_daily * 0.1, 365)
        consumption += noise
        
        # Create dataframe
        df = pd.DataFrame({
            'Date': dates,
            'Energy_Consumption_kWh': np.round(consumption, 2),
            'Source': 'Survey Data'
        })
        
        # Add temperature if available in survey
        if 'location' in user_data:
            # Simple temperature model based on location
            if 'hot' in user_data['location'].lower():
                temp = 25 + 10 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 5, 365)
            else:
                temp = 15 + 10 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 5, 365)
            df['Temperature_C'] = np.round(temp, 1)
        
        return df
    return None

# Option 2: Upload Custom Data
def parse_uploaded_file(uploaded_file):
    """Parse uploaded Excel/CSV file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        return df, None
    except Exception as e:
        return None, str(e)

# Main content layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="option-card">
        <h3>📋 Use Survey Data</h3>
        <p>Use the data from your completed energy survey</p>
        <p><small>Automatically generates realistic consumption patterns based on your survey responses</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Load Survey Data", key="load_survey", use_container_width=True):
        if 'survey_completed' in st.session_state and st.session_state.survey_completed:
            with st.spinner("Generating data from survey..."):
                df = load_survey_data()
                if df is not None:
                    st.session_state.forecast_data = df
                    st.session_state.data_source = "survey"
                    st.session_state.data_loaded = True
                    st.success("✅ Survey data loaded successfully!")
                    st.rerun()
                else:
                    st.error("No survey data found. Please complete the survey first.")
        else:
            st.warning("⚠️ No completed survey found.")
            
            if st.button("Take Survey Now", key="goto_survey"):
                st.switch_page("pages/3_Survey.py")

with col2:
    st.markdown("""
    <div class="option-card">
        <h3>📁 Upload Your Data</h3>
        <p>Upload your own energy consumption data</p>
        <p><small>Supports Excel (.xlsx, .xls) and CSV formats</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['xlsx', 'xls', 'csv'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.info(f"📄 **File:** {uploaded_file.name}")
        
        # Parse the file
        df, error = parse_uploaded_file(uploaded_file)
        
        if error:
            st.error(f"❌ Error: {error}")
        elif df is not None:
            # Show preview
            st.markdown("#### Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            st.markdown("#### Column Mapping")
            
            # Auto-detect columns
            date_columns = [col for col in df.columns if any(keyword in col.lower() 
                                                           for keyword in ['date', 'time', 'day', 'timestamp'])]
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Date column
                date_col = st.selectbox(
                    "Date Column",
                    options=['Select...'] + df.columns.tolist(),
                    index=date_columns[0] + 1 if date_columns else 1
                )
            
            with col_b:
                # Consumption column
                consumption_col = st.selectbox(
                    "Consumption Column (kWh)",
                    options=['Select...'] + numeric_columns,
                    index=numeric_columns[0] + 1 if numeric_columns else 1
                )
            
            if st.button("✅ Use This Data", type="primary", use_container_width=True,
                        disabled=(date_col == 'Select...' or consumption_col == 'Select...')):
                try:
                    # Clean and prepare data
                    df_clean = df.copy()
                    df_clean['Date'] = pd.to_datetime(df_clean[date_col])
                    df_clean['Energy_Consumption_kWh'] = pd.to_numeric(df_clean[consumption_col], errors='coerce')
                    
                    # Drop invalid rows
                    df_clean = df_clean.dropna(subset=['Date', 'Energy_Consumption_kWh'])
                    
                    # Sort by date
                    df_clean = df_clean.sort_values('Date').reset_index(drop=True)
                    
                    # Add source info
                    df_clean['Source'] = 'Uploaded Data'
                    
                    # Store in session state
                    st.session_state.forecast_data = df_clean
                    st.session_state.data_source = "uploaded"
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ Data loaded successfully! {len(df_clean)} records")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")

# Option 3: Use Sample Data
st.markdown("---")
st.markdown("### 🧪 Try with Sample Data")

st.markdown("""
<div class="option-card">
    <h3>📊 Use Sample Data</h3>
    <p>Try the forecasting features with pre-generated sample data</p>
    <p><small>1 year of realistic energy consumption patterns with seasonality</small></p>
</div>
""", unsafe_allow_html=True)

if st.button("Use Sample Data", key="use_sample", use_container_width=True):
    with st.spinner("Generating sample data..."):
        # Generate realistic sample data
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        
        # Base pattern
        base_daily = 30  # Average 30 kWh per day
        
        # Seasonality
        seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365)
        
        # Weekly pattern
        weekly = np.where(dates.weekday < 5, 5, -5)
        
        # Trend
        trend = np.linspace(0, 5, 365)
        
        # Noise
        noise = np.random.normal(0, 3, 365)
        
        consumption = base_daily + seasonal + weekly + trend + noise
        consumption = np.maximum(consumption, 15)  # Minimum
        
        # Temperature
        temp = 15 + 20 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 5, 365)
        
        df = pd.DataFrame({
            'Date': dates,
            'Energy_Consumption_kWh': np.round(consumption, 2),
            'Temperature_C': np.round(temp, 1),
            'Source': 'Sample Data'
        })
        
        st.session_state.forecast_data = df
        st.session_state.data_source = "sample"
        st.session_state.data_loaded = True
        
        st.success("✅ Sample data generated!")
        st.rerun()

# Navigation
st.markdown("---")
if st.session_state.data_loaded and st.session_state.forecast_data is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Go to Forecasting", type="primary", use_container_width=True):
            st.switch_page("pages/forecast.py")

# Display current status
st.markdown("---")
st.markdown("### 📈 Current Status")

if st.session_state.data_loaded:
    data = st.session_state.forecast_data
    st.success(f"✅ Data loaded from {st.session_state.data_source}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Records", f"{len(data):,}")
    
    with col2:
        date_range = f"{data['Date'].min().strftime('%b %d, %Y')} to {data['Date'].max().strftime('%b %d, %Y')}"
        st.metric("Date Range", date_range)
    
    with col3:
        avg_consumption = data['Energy_Consumption_kWh'].mean()
        st.metric("Avg Daily", f"{avg_consumption:.1f} kWh")
    
    # Quick preview
    with st.expander("📋 View Data Preview"):
        st.dataframe(data.head(10), use_container_width=True)
        
        # Basic statistics
        st.markdown("#### 📊 Basic Statistics")
        stats = data['Energy_Consumption_kWh'].describe()
        st.write(stats)
else:
    st.info("👈 Select a data source option above")

# Footer
st.markdown("---")
st.caption("💡 Tip: Complete the survey for personalized data or upload your own historical data for accurate forecasts")
