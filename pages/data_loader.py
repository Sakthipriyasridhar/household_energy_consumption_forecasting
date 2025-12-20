# pages/2_Data_Loader.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Load Energy Data",
    page_icon="📊",
    layout="wide"
)

# Add this CSS right after your existing CSS section (around line 120)

# Replace or add to your existing CSS:
st.markdown("""
<style>
    /* Original CSS... */
    .option-card {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px solid #e0e0e0;
        transition: all 0.3s;
    }
    /* ... keep all your existing CSS ... */
    
    /* NEW CSS FOR BIGGER TABS - Add this at the end */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        margin-bottom: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        padding: 0 1.5rem;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(to bottom, #f8f9fa, #e9ecef);
        border-radius: 12px 12px 0 0;
        border: 2px solid #dee2e6;
        border-bottom: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(to bottom, #2E86AB, #1b6b91) !important;
        color: white !important;
        border-color: #2E86AB !important;
        box-shadow: 0 4px 8px rgba(46, 134, 171, 0.3);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(to bottom, #e9ecef, #dee2e6);
        transform: translateY(-1px);
    }
    
    .stTabs [aria-selected="true"]:hover {
        background: linear-gradient(to bottom, #257293, #155673) !important;
    }
    
    /* Tab content with shadow */
    .stTabs [data-baseweb="tab-panel"] {
        padding: 1.5rem;
        border: 2px solid #2E86AB;
        border-top: none;
        border-radius: 0 0 12px 12px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-top: -1px;
    }
    
    /* Make tab icons bigger */
    .stTabs [data-baseweb="tab"] span {
        font-size: 1.3rem;
        margin-right: 0.5rem;
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
if 'data_quality_report' not in st.session_state:
    st.session_state.data_quality_report = {}
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Upload Your Data"  # Default tab

# Function to detect date columns intelligently
def detect_date_columns(df):
    """Automatically detect date/time columns in the dataframe"""
    date_columns = []
    
    for col in df.columns:
        col_str = str(col).lower()
        
        # Check column name for date/time keywords
        date_keywords = ['date', 'time', 'day', 'timestamp', 'dt', 'datetime', 'period']
        if any(keyword in col_str for keyword in date_keywords):
            date_columns.append(col)
            continue
        
        # Check data type
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
            continue
        
        # Try to parse first few non-null values
        sample_data = df[col].dropna().head(5)
        if len(sample_data) > 0:
            try:
                # Try to convert to datetime
                pd.to_datetime(sample_data)
                date_columns.append(col)
            except:
                pass
    
    return list(set(date_columns))

# Function to detect numeric columns (for consumption)
def detect_numeric_columns(df):
    """Detect numeric columns for energy consumption"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Also look for columns with numeric names
    energy_keywords = ['kwh', 'consumption', 'usage', 'energy', 'power', 'load', 'demand', 'value']
    for col in df.columns:
        col_str = str(col).lower()
        if any(keyword in col_str for keyword in energy_keywords):
            # Try to convert to numeric
            try:
                pd.to_numeric(df[col], errors='coerce')
                if col not in numeric_cols:
                    numeric_cols.append(col)
            except:
                pass
    
    return numeric_cols

# Function to analyze data quality
def analyze_data_quality(df, date_col, consumption_col):
    """Analyze data quality and missing values"""
    report = {
        'total_rows': len(df),
        'date_missing': 0,
        'consumption_missing': 0,
        'date_range': None,
        'consumption_stats': {},
        'quality_score': 100,
        'issues': []
    }
    
    # Check missing values
    if date_col and date_col in df.columns:
        date_missing = df[date_col].isnull().sum()
        report['date_missing'] = int(date_missing)
        if date_missing > 0:
            report['quality_score'] -= 30
            report['issues'].append(f"Missing dates: {int(date_missing)} rows")
    
    if consumption_col and consumption_col in df.columns:
        consumption_missing = df[consumption_col].isnull().sum()
        report['consumption_missing'] = int(consumption_missing)
        if consumption_missing > 0:
            report['quality_score'] -= 30
            report['issues'].append(f"Missing consumption values: {int(consumption_missing)} rows")
    
    # Check date range
    if date_col and date_col in df.columns and df[date_col].notna().any():
        valid_dates = df[date_col].dropna()
        if len(valid_dates) > 0:
            try:
                dates_converted = pd.to_datetime(valid_dates)
                report['date_range'] = {
                    'start': dates_converted.min(),
                    'end': dates_converted.max(),
                    'days': (dates_converted.max() - dates_converted.min()).days
                }
            except Exception as e:
                report['issues'].append(f"Date column contains invalid date formats: {str(e)}")
    
    # Check consumption statistics
    if consumption_col and consumption_col in df.columns and df[consumption_col].notna().any():
        try:
            valid_consumption = pd.to_numeric(df[consumption_col], errors='coerce')
            report['consumption_stats'] = {
                'mean': float(valid_consumption.mean()),
                'min': float(valid_consumption.min()),
                'max': float(valid_consumption.max()),
                'std': float(valid_consumption.std())
            }
            
            # Check for zeros or negative values (if not expected)
            negative_count = (valid_consumption < 0).sum()
            if negative_count > 0:
                report['issues'].append(f"Negative consumption values: {int(negative_count)} rows")
        except Exception as e:
            report['issues'].append(f"Error analyzing consumption data: {str(e)}")
    
    # Check for duplicates
    if date_col and date_col in df.columns:
        duplicates = df.duplicated(subset=[date_col]).sum()
        if duplicates > 0:
            report['issues'].append(f"Duplicate dates: {int(duplicates)} rows")
            report['quality_score'] -= 10
    
    # Ensure quality score is between 0 and 100
    report['quality_score'] = max(0, min(100, report['quality_score']))
    
    return report

# Function to clean and prepare data
def clean_and_prepare_data(df, date_col, consumption_col, location_col=None, fill_method='interpolate'):
    """Clean and prepare the data for analysis"""
    df_clean = df.copy()
    
    try:
        # 1. Handle date column
        if date_col and date_col in df_clean.columns:
            df_clean['Date'] = pd.to_datetime(df_clean[date_col], errors='coerce')
        else:
            # Create dummy dates if no date column
            st.warning("No valid date column found. Creating sequential dates starting from today.")
            df_clean['Date'] = pd.date_range(start=datetime.now(), periods=len(df_clean), freq='D')
        
        # 2. Handle consumption column
        if consumption_col and consumption_col in df_clean.columns:
            df_clean['Energy_Consumption_kWh'] = pd.to_numeric(df_clean[consumption_col], errors='coerce')
        else:
            st.error("No valid consumption column found. Please select a numeric column.")
            return None, None
        
        # 3. Handle location column
        if location_col and location_col in df_clean.columns:
            df_clean['Location'] = df_clean[location_col].astype(str)
        elif 'Location' not in df_clean.columns:
            df_clean['Location'] = 'Unknown'
        
        # 4. Handle missing values in date column
        date_missing = df_clean['Date'].isnull().sum()
        if date_missing > 0:
            st.warning(f"Found {int(date_missing)} rows with invalid dates. These rows will be dropped.")
            df_clean = df_clean.dropna(subset=['Date'])
        
        # 5. Handle missing values in consumption column
        consumption_missing_before = df_clean['Energy_Consumption_kWh'].isnull().sum()
        
        if consumption_missing_before > 0:
            st.info(f"Found {int(consumption_missing_before)} rows with missing consumption values.")
            
            if fill_method == 'interpolate' and len(df_clean) > 1:
                # Sort by date first
                df_clean = df_clean.sort_values('Date')
                # Interpolate missing values
                df_clean['Energy_Consumption_kWh'] = df_clean['Energy_Consumption_kWh'].interpolate(method='linear')
                # Fill any remaining NaNs with forward/backward fill
                df_clean['Energy_Consumption_kWh'] = df_clean['Energy_Consumption_kWh'].fillna(method='ffill').fillna(method='bfill')
                st.success("Missing values filled using interpolation")
            
            elif fill_method == 'mean' and len(df_clean) > 0:
                mean_value = df_clean['Energy_Consumption_kWh'].mean()
                if pd.notna(mean_value):
                    df_clean['Energy_Consumption_kWh'] = df_clean['Energy_Consumption_kWh'].fillna(mean_value)
                    st.success(f"Missing values filled with mean: {mean_value:.2f} kWh")
                else:
                    df_clean['Energy_Consumption_kWh'] = df_clean['Energy_Consumption_kWh'].fillna(0)
                    st.warning("Could not calculate mean, filled missing values with 0")
            
            elif fill_method == 'drop':
                df_clean = df_clean.dropna(subset=['Energy_Consumption_kWh'])
                st.success(f"Dropped {int(consumption_missing_before)} rows with missing consumption values")
        
        # 6. Sort by date
        df_clean = df_clean.sort_values('Date').reset_index(drop=True)
        
        # 7. Add source info
        df_clean['Source'] = 'Uploaded Data'
        df_clean['Original_Date_Column'] = date_col if date_col else 'Generated'
        df_clean['Original_Consumption_Column'] = consumption_col if consumption_col else 'Generated'
        
        # 8. Calculate quality metrics
        consumption_missing_after = df_clean['Energy_Consumption_kWh'].isnull().sum()
        
        quality_report = {
            'original_rows': len(df),
            'cleaned_rows': len(df_clean),
            'date_missing_removed': int(date_missing),
            'consumption_missing_filled': int(consumption_missing_before - consumption_missing_after),
            'consumption_missing_remaining': int(consumption_missing_after),
            'date_range': {
                'start': df_clean['Date'].min(),
                'end': df_clean['Date'].max(),
                'days': (df_clean['Date'].max() - df_clean['Date'].min()).days
            } if len(df_clean) > 0 else None,
            'fill_method_used': fill_method
        }
        
        return df_clean, quality_report
        
    except Exception as e:
        st.error(f"Error cleaning data: {str(e)}")
        return None, None

# Function to parse uploaded file with intelligent detection
def parse_uploaded_file(uploaded_file):
    """Parse uploaded file with intelligent column detection"""
    try:
        # Determine file type and read
        if uploaded_file.name.endswith('.csv'):
            # Try different encodings for CSV
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
        else:
            # For Excel files
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
        
        return df, None
    except Exception as e:
        return None, str(e)

# Function to generate survey-based data
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
        end_date = start_date + timedelta(days=365)
    
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Get survey data
    user_data = st.session_state.user_data
    
    # Extract key parameters from survey
    monthly_usage = user_data.get('monthly_consumption', 900)
    if monthly_usage == 0:
        monthly_usage = 900
    
    location = user_data.get('household', {}).get('location', 'Chennai')
    num_appliances = len(user_data.get('appliances', []))
    
    # Calculate daily average
    daily_avg = monthly_usage / 30
    
    # Generate consumption pattern
    consumption = []
    temperatures = []
    
    for i, date in enumerate(dates):
        # Base consumption
        base = daily_avg
        
        # Day of week pattern
        day_of_week = date.weekday()
        if day_of_week < 5:
            base *= 1.15
        else:
            base *= 0.85
        
        # Seasonality
        month = date.month
        day_of_year = date.dayofyear
        
        if location.lower() in ['chennai', 'madurai', 'hyderabad']:
            if month in [3, 4, 5, 6]:
                seasonal_factor = 1.4
            elif month in [7, 8, 9]:
                seasonal_factor = 1.2
            elif month in [10, 11]:
                seasonal_factor = 1.0
            else:
                seasonal_factor = 0.9
        else:
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        base *= seasonal_factor
        
        # Appliance factor
        appliance_factor = 1 + (num_appliances * 0.03)
        base *= appliance_factor
        
        # Random variation
        random_variation = np.random.normal(0, base * 0.15)
        final_consumption = max(base + random_variation, daily_avg * 0.5)
        
        # Temperature
        if location.lower() in ['chennai', 'madurai', 'hyderabad']:
            base_temp = 28 + 8 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
        elif location.lower() in ['bangalore', 'coimbatore']:
            base_temp = 22 + 6 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
        else:
            base_temp = 25 + 7 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
        
        temp = base_temp + np.random.normal(0, 3)
        temp = max(min(temp, 45), 10)
        
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

# Function to generate sample data
def generate_sample_data(start_date=None, end_date=None, months=12):
    """Generate realistic sample data with user-defined date range"""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    
    if end_date is None:
        if isinstance(months, int):
            end_date = start_date + timedelta(days=months*30)
        else:
            end_date = start_date + timedelta(days=365)
    
    # Ensure end_date is after start_date
    if end_date <= start_date:
        end_date = start_date + timedelta(days=365)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base pattern
    base_daily = 30
    
    # Complex seasonality
    seasonal = 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    
    # Weekly pattern
    weekly = np.where(dates.weekday < 5, 4, -4)
    
    # Trend
    trend = np.linspace(0, 3, len(dates))
    
    # Holidays
    holidays = np.zeros(len(dates))
    holiday_indices = np.random.choice(len(dates), size=min(10, len(dates)//10), replace=False)
    holidays[holiday_indices] = -10
    
    # Noise
    noise = np.random.normal(0, 2.5, len(dates))
    
    consumption = base_daily + seasonal + weekly + trend + holidays + noise
    consumption = np.maximum(consumption, 15)
    
    # Temperature
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
        'Household_Size': 4,
        'Sample_Period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    })
    
    st.session_state.prediction_settings.update({
        'start_date': start_date,
        'end_date': end_date,
        'prediction_months': months,
        'data_type': 'sample'
    })
    
    return df

# Main interface
st.markdown("### 📋 Select Data Source")

# Create tabs for different data sources
tab1, tab2, tab3 = st.tabs(["📊 Use Survey Data", "📁 Upload Your Data", "🧪 Try Sample Data"])

# Initialize variables to control what's shown
show_survey_data = False
show_upload_data = False
show_sample_data = False

with tab1:
    # Track that we're in survey tab
    show_survey_data = True
    
    st.markdown("""
    <div class="option-card">
        <h3>📋 Generate Data from Survey</h3>
        <p>Create realistic energy consumption patterns based on your survey responses</p>
        <p><small>Customize the time period and generate personalized data</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'survey_completed' in st.session_state and st.session_state.survey_completed:
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
                "Months to Generate",
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
                    st.session_state.data_quality_report = {
                        'data_type': 'survey',
                        'date_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
                        'total_records': len(df)
                    }
                    st.success(f"✅ Generated {len(df)} days of personalized data!")
                    st.rerun()
                else:
                    st.error("Could not generate data. Please check your survey responses.")
    
    else:
        st.warning("⚠️ No completed survey found.")
        if st.button("Take Survey Now", type="secondary"):
            st.switch_page("pages/survey.py")

with tab2:
    # Track that we're in upload tab
    show_upload_data = True
    
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
        with st.spinner("Analyzing uploaded file..."):
            df, error = parse_uploaded_file(uploaded_file)
        
        if error:
            st.error(f"❌ Error reading file: {error}")
        elif df is not None:
            # Show file info
            st.markdown("#### 📋 File Information")
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Rows", len(df))
            with col_info2:
                st.metric("Columns", len(df.columns))
            with col_info3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Show preview
            st.markdown("#### 👀 Data Preview (First 10 rows)")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Show column types
            st.markdown("#### 📝 Column Types")
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.notnull().sum().values,
                'Null %': (df.isnull().sum().values / len(df) * 100).round(1)
            })
            st.dataframe(dtype_df, use_container_width=True)
            
            st.markdown("#### 🎯 Intelligent Column Detection")
            
            # Auto-detect columns
            date_columns = detect_date_columns(df)
            numeric_columns = detect_numeric_columns(df)
            all_columns = df.columns.tolist()
            
            # Create column mapping interface
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown("**Date/Time Column**")
                if date_columns:
                    date_col = st.selectbox(
                        "Select date column",
                        options=['Auto-detect'] + all_columns,
                        index=0,
                        help="Column containing dates or timestamps",
                        key="date_col_select"
                    )
                    if date_col == 'Auto-detect':
                        date_col = date_columns[0] if date_columns else None
                        if date_col:
                            st.success(f"✅ Auto-detected: {date_col}")
                else:
                    st.warning("No date columns detected")
                    date_col = st.selectbox(
                        "Select date column",
                        options=['None found'] + all_columns,
                        help="Manually select column containing dates",
                        key="date_col_manual"
                    )
                    if date_col == 'None found':
                        date_col = None
            
            with col_b:
                st.markdown("**Energy Consumption Column**")
                if numeric_columns:
                    consumption_col = st.selectbox(
                        "Select consumption column",
                        options=['Auto-detect'] + all_columns,
                        index=0,
                        help="Column containing energy consumption values (kWh)",
                        key="consumption_col_select"
                    )
                    if consumption_col == 'Auto-detect':
                        consumption_col = numeric_columns[0] if numeric_columns else None
                        if consumption_col:
                            st.success(f"✅ Auto-detected: {consumption_col}")
                else:
                    st.warning("No numeric columns detected")
                    consumption_col = st.selectbox(
                        "Select consumption column",
                        options=['None found'] + all_columns,
                        help="Manually select numeric column",
                        key="consumption_col_manual"
                    )
                    if consumption_col == 'None found':
                        consumption_col = None
            
            with col_c:
                st.markdown("**Location Column (Optional)**")
                location_col = st.selectbox(
                    "Select location column",
                    options=['Not specified'] + all_columns,
                    help="Column containing location information",
                    key="location_col_select"
                )
                if location_col == 'Not specified':
                    location_col = None
            
            # Data quality analysis
            if date_col and consumption_col:
                st.markdown("#### 🔍 Data Quality Analysis")
                
                with st.spinner("Analyzing data quality..."):
                    quality_report = analyze_data_quality(df, date_col, consumption_col)
                    st.session_state.data_quality_report = quality_report
                
                # Display quality metrics
                col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                
                with col_q1:
                    quality_score = quality_report.get('quality_score', 0)
                    if quality_score >= 80:
                        st.metric("Quality Score", f"{quality_score}%", "Good", delta_color="normal")
                    elif quality_score >= 60:
                        st.metric("Quality Score", f"{quality_score}%", "Fair", delta_color="off")
                    else:
                        st.metric("Quality Score", f"{quality_score}%", "Poor", delta_color="inverse")
                
                with col_q2:
                    missing_dates = quality_report.get('date_missing', 0)
                    st.metric("Missing Dates", missing_dates)
                
                with col_q3:
                    missing_consumption = quality_report.get('consumption_missing', 0)
                    st.metric("Missing Values", missing_consumption)
                
                with col_q4:
                    if quality_report.get('date_range'):
                        days = quality_report['date_range']['days']
                        st.metric("Date Range (days)", days)
                
                # Show issues
                if quality_report.get('issues'):
                    st.markdown("#### ⚠️ Data Issues Found")
                    for issue in quality_report['issues']:
                        st.warning(issue)
                
                # Missing value handling options
                missing_consumption = quality_report.get('consumption_missing', 0)
                missing_dates = quality_report.get('date_missing', 0)
                
                if missing_consumption > 0 or missing_dates > 0:
                    st.markdown("#### 🛠️ Handle Missing Values")
                    
                    fill_method = st.radio(
                        "How to handle missing consumption values?",
                        options=['interpolate', 'mean', 'drop'],
                        format_func=lambda x: {
                            'interpolate': 'Interpolate (recommended for time series)',
                            'mean': 'Fill with mean value',
                            'drop': 'Drop rows with missing values'
                        }[x],
                        horizontal=True
                    )
                    
                    st.info(f"**{missing_consumption} missing consumption values** will be handled using: **{fill_method}**")
                else:
                    fill_method = 'interpolate'  # Default
                
                # Additional settings
                st.markdown("#### ⚙️ Forecasting Settings")
                
                col_set1, col_set2 = st.columns(2)
                
                with col_set1:
                    future_months = st.number_input(
                        "Months to Forecast",
                        min_value=1,
                        max_value=24,
                        value=6,
                        help="How many months into the future to forecast",
                        key="future_months_upload"
                    )
                
                with col_set2:
                    if quality_report.get('date_range'):
                        start_dt = quality_report['date_range']['start']
                        end_dt = quality_report['date_range']['end']
                        if start_dt and end_dt:
                            st.info(f"**Data Range:** {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
                
                # Process button
                if st.button("✅ Process and Clean Data", type="primary", use_container_width=True, key="process_upload"):
                    with st.spinner("Cleaning and preparing data..."):
                        result = clean_and_prepare_data(
                            df, 
                            date_col, 
                            consumption_col, 
                            location_col,
                            fill_method=fill_method
                        )
                        
                        if result and result[0] is not None:
                            df_clean, cleaning_report = result
                            
                            # Store in session state
                            st.session_state.forecast_data = df_clean
                            st.session_state.data_loaded = True
                            st.session_state.prediction_settings.update({
                                'prediction_months': future_months,
                                'start_date': df_clean['Date'].max() if len(df_clean) > 0 else None,
                                'data_type': 'uploaded',
                                'cleaning_report': cleaning_report
                            })
                            
                            # Show cleaning results
                            st.success(f"✅ Data cleaned successfully!")
                            st.markdown("#### 🧹 Cleaning Results")
                            
                            col_c1, col_c2, col_c3 = st.columns(3)
                            with col_c1:
                                st.metric("Original Rows", cleaning_report.get('original_rows', 0))
                            with col_c2:
                                st.metric("Cleaned Rows", cleaning_report.get('cleaned_rows', 0))
                            with col_c3:
                                filled = cleaning_report.get('consumption_missing_filled', 0)
                                st.metric("Values Filled", filled)
                            
                            st.rerun()
                        else:
                            st.error("Failed to clean data. Please check your column selections.")
            else:
                st.error("Please select both date and consumption columns to proceed.")
    else:
        st.info("👆 Upload a file to get started")

with tab3:
    # Track that we're in sample tab
    show_sample_data = True
    
    st.markdown("""
    <div class="option-card">
        <h3>🧪 Generate Sample Data</h3>
        <p>Try the forecasting features with realistic sample data</p>
        <p><small>Perfect for testing and demonstration purposes</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Date range selection for sample data
    st.markdown("#### 📅 Select Date Range for Sample Data")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        sample_start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now(),
            key="sample_start"
        )
    
    with col_s2:
        sample_end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            min_value=sample_start_date + timedelta(days=30),
            key="sample_end"
        )
    
    with col_s3:
        sample_months = st.number_input(
            "Or specify months",
            min_value=1,
            max_value=36,
            value=12,
            key="sample_months",
            help="Generate data for this many months from start date"
        )
        
        use_months = st.checkbox("Use months instead of end date", value=False)
    
    # Data complexity options
    st.markdown("#### ⚙️ Sample Data Options")
    
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        include_temperature = st.checkbox("Include temperature data", value=True)
        include_seasonality = st.checkbox("Include seasonality patterns", value=True)
    
    with col_opt2:
        include_trend = st.checkbox("Include upward trend", value=True)
        noise_level = st.slider("Noise level", 0.0, 1.0, 0.3)
    
    if st.button("🧪 Generate Sample Data", type="primary", use_container_width=True, key="generate_sample"):
        with st.spinner("Generating realistic sample data..."):
            if use_months:
                df = generate_sample_data(
                    start_date=sample_start_date,
                    months=sample_months
                )
            else:
                df = generate_sample_data(
                    start_date=sample_start_date,
                    end_date=sample_end_date
                )
            
            # Apply user options
            if not include_temperature and 'Temperature_C' in df.columns:
                df = df.drop(columns=['Temperature_C'])
            
            if not include_seasonality:
                # Remove seasonality by adjusting values
                df['Energy_Consumption_kWh'] = df['Energy_Consumption_kWh'] * 0.9 + 25
            
            if not include_trend:
                # Remove trend
                df['Energy_Consumption_kWh'] = df['Energy_Consumption_kWh'] - np.linspace(0, 3, len(df))
            
            # Adjust noise level
            if noise_level != 0.3:
                current_std = df['Energy_Consumption_kWh'].std()
                target_std = 2.5 * noise_level / 0.3
                scale_factor = target_std / current_std if current_std > 0 else 1
                df['Energy_Consumption_kWh'] = df['Energy_Consumption_kWh'] * scale_factor
            
            st.session_state.forecast_data = df
            st.session_state.data_loaded = True
            st.session_state.data_quality_report = {
                'data_type': 'sample',
                'date_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
                'total_records': len(df),
                'options_used': {
                    'include_temperature': include_temperature,
                    'include_seasonality': include_seasonality,
                    'include_trend': include_trend,
                    'noise_level': noise_level
                }
            }
            
            st.success(f"✅ Generated {len(df)} days of sample data!")
            st.info(f"📅 **Date Range:** {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
            st.rerun()

# Data Preview Section - Only show if data is loaded from the CURRENT tab
st.markdown("---")

# Only show data status if we have loaded data AND it matches the current context
if st.session_state.data_loaded and st.session_state.forecast_data is not None:
    data = st.session_state.forecast_data
    current_data_type = st.session_state.prediction_settings.get('data_type', '')
    
    # Check if we should show the data based on current tab
    should_show_data = False
    
    if show_survey_data and current_data_type == 'survey':
        should_show_data = True
    elif show_upload_data and current_data_type == 'uploaded':
        should_show_data = True
    elif show_sample_data and current_data_type == 'sample':
        should_show_data = True
    # Also show if no specific tab is active (initial load)
    elif not (show_survey_data or show_upload_data or show_sample_data):
        should_show_data = True
    
    if should_show_data:
        st.markdown("### 📈 Current Data Status")
        
        # Display data source info
        data_source_map = {
            'survey': 'Survey Data',
            'uploaded': 'Uploaded Data',
            'sample': 'Sample Data'
        }
        data_source_name = data_source_map.get(current_data_type, 'Unknown')
        st.markdown(f"#### 📋 Data Source: **{data_source_name}**")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(data):,}")
        
        with col2:
            if 'Date' in data.columns and len(data) > 0:
                date_range = f"{data['Date'].min().strftime('%b %d, %Y')} to {data['Date'].max().strftime('%b %d, %Y')}"
                st.metric("Date Range", date_range)
            else:
                st.metric("Date Range", "N/A")
        
        with col3:
            if 'Energy_Consumption_kWh' in data.columns:
                avg_consumption = data['Energy_Consumption_kWh'].mean()
                st.metric("Avg Daily", f"{avg_consumption:.1f} kWh")
            else:
                st.metric("Avg Daily", "N/A")
        
        with col4:
            if 'Date' in data.columns and len(data) > 0:
                total_days = (data['Date'].max() - data['Date'].min()).days + 1
                st.metric("Total Days", total_days)
            else:
                st.metric("Total Days", "N/A")
        
        # Data quality summary if available
        if st.session_state.data_quality_report and st.session_state.data_quality_report.get('data_type') == current_data_type:
            st.markdown("#### 🔍 Data Quality Summary")
            
            quality_cols = st.columns(4)
            with quality_cols[0]:
                if 'quality_score' in st.session_state.data_quality_report:
                    score = st.session_state.data_quality_report['quality_score']
                    st.progress(score/100, text=f"Quality: {score}%")
            
            with quality_cols[1]:
                if 'consumption_missing' in st.session_state.data_quality_report:
                    missing = st.session_state.data_quality_report['consumption_missing']
                    if missing == 0:
                        st.success("✅ No missing values")
                    else:
                        st.warning(f"⚠️ {missing} missing values")
            
            with quality_cols[2]:
                if 'date_missing' in st.session_state.data_quality_report:
                    missing_dates = st.session_state.data_quality_report['date_missing']
                    if missing_dates == 0:
                        st.success("✅ Valid dates")
                    else:
                        st.warning(f"⚠️ {missing_dates} invalid dates")
            
            with quality_cols[3]:
                if 'issues' in st.session_state.data_quality_report:
                    issues = len(st.session_state.data_quality_report['issues'])
                    if issues == 0:
                        st.success("✅ No issues")
                    else:
                        st.warning(f"⚠️ {issues} issues found")
        
        # Data preview in tabs
        tab_preview, tab_stats, tab_viz, tab_missing = st.tabs(["📋 Data Preview", "📊 Statistics", "📈 Visualization", "🔍 Missing Data"])
        
        with tab_preview:
            # Show first 20 rows
            st.dataframe(data.head(20), use_container_width=True)
            
            # Show column info
            st.markdown("#### 📝 Column Information")
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes.astype(str),
                'Non-Null': data.notnull().sum().values,
                'Null %': (data.isnull().sum().values / len(data) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
        
        with tab_stats:
            # Basic statistics
            st.markdown("#### 📊 Consumption Statistics")
            if 'Energy_Consumption_kWh' in data.columns:
                stats = data['Energy_Consumption_kWh'].describe()
                st.write(stats)
            
            # Monthly statistics
            if len(data) > 30 and 'Date' in data.columns:
                data_monthly = data.copy()
                data_monthly['YearMonth'] = data_monthly['Date'].dt.to_period('M')
                monthly_stats = data_monthly.groupby('YearMonth')['Energy_Consumption_kWh'].agg(['mean', 'min', 'max', 'sum', 'std'])
                st.markdown("#### 📅 Monthly Statistics")
                st.dataframe(monthly_stats, use_container_width=True)
        
        with tab_viz:
            # Create time series plot
            if 'Date' in data.columns and 'Energy_Consumption_kWh' in data.columns:
                fig = go.Figure()
                
                # Add consumption line
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['Energy_Consumption_kWh'],
                    mode='lines',
                    name='Energy Consumption',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Consumption: %{y:.1f} kWh<extra></extra>'
                ))
                
                # Add 7-day moving average if enough data
                if len(data) > 7:
                    data_sorted = data.sort_values('Date')
                    moving_avg = data_sorted['Energy_Consumption_kWh'].rolling(window=7, min_periods=1).mean()
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
                    showlegend=True,
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No date or consumption data available for visualization")
        
        with tab_missing:
            # Show missing data analysis
            st.markdown("#### 🔍 Missing Data Analysis")
            
            # Calculate missing values per column
            missing_data = data.isnull().sum()
            missing_percentage = (missing_data / len(data) * 100).round(2)
            
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Values': missing_data.values,
                'Percentage': missing_percentage.values
            })
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
                
                # Visualize missing data
                fig_missing = go.Figure(data=[
                    go.Bar(
                        x=missing_df['Column'],
                        y=missing_df['Percentage'],
                        text=missing_df['Percentage'].apply(lambda x: f'{x}%'),
                        textposition='auto',
                        marker_color='#ff6b6b'
                    )
                ])
                
                fig_missing.update_layout(
                    title='Percentage of Missing Values by Column',
                    xaxis_title='Column',
                    yaxis_title='Missing %',
                    height=400
                )
                
                st.plotly_chart(fig_missing, use_container_width=True)
                
                st.warning("⚠️ Missing data detected. Consider filling or removing these values for better forecasting.")
            else:
                st.success("✅ No missing data found in the current dataset!")
        
        # Navigation and export section
        st.markdown("---")
        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
        
        with col_nav2:
            if st.button("🚀 Proceed to Forecasting", type="primary", use_container_width=True, key="goto_forecast"):
                # Store prediction settings
                if 'prediction_settings' in st.session_state:
                    st.session_state.prediction_months = st.session_state.prediction_settings.get('prediction_months', 12)
                
                st.switch_page("pages/forecast.py")
        
        # Data export option - FIXED Excel export error
        st.markdown("### 💾 Export Options")
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            csv_data = data.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"energy_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download the current dataset as CSV",
                use_container_width=True
            )
        
        with export_col2:
            try:
                # Create Excel file in memory
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    data.to_excel(writer, index=False, sheet_name='EnergyData')
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 Download as Excel",
                    data=excel_data,
                    file_name=f"energy_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download the current dataset as Excel",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error creating Excel file: {str(e)}")
                st.info("Try downloading as CSV instead")
    
    else:
        # Show message that data exists but from different source
        st.info(f"📊 Data loaded from **{data_source_map.get(current_data_type, 'another source')}**. Switch to the appropriate tab to view it.")

else:
    # No data loaded at all
    st.info("👈 Select a data source option above to begin")
    
    # Quick tips
    st.markdown("---")
    st.markdown("### 💡 Quick Tips for Data Upload")
    
    tips_col1, tips_col2, tips_col3 = st.columns(3)
    
    with tips_col1:
        st.markdown("""
        **📋 Supported Formats**
        - CSV files (comma or semicolon separated)
        - Excel files (.xlsx, .xls)
        - Google Sheets (download as CSV)
        """)
    
    with tips_col2:
        st.markdown("""
        **📊 Column Requirements**
        - **Date Column:** Any date/time format
        - **Consumption Column:** Numeric values (kWh)
        - **Optional:** Location, temperature, etc.
        """)
    
    with tips_col3:
        st.markdown("""
        **🔍 Data Quality**
        - Missing values auto-detected
        - Multiple filling options
        - Data quality scoring
        - Issue reporting
        """)

# Footer
st.markdown("---")
st.caption("💡 **Tip:** For best results, upload at least 1 year of historical daily or monthly data. The system automatically detects and handles common data issues.")
