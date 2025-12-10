import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Time Series Models (Optional)
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.info("⚠️ Statsmodels not installed. Classical models (ARIMA) won't be available.")

# Page Configuration
st.set_page_config(
    page_title="AI Forecast - Energy Optimizer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
def load_css():
    css = """
    <style>
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Fix sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e, #16213e);
        }
        
        .ml-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid;
            margin-bottom: 1rem;
        }
        
        .rf-card { border-left-color: #4CAF50; }
        .xgb-card { border-left-color: #FF6B6B; }
        .lgb-card { border-left-color: #00BCD4; }
        .lr-card { border-left-color: #9C27B0; }
        .arima-card { border-left-color: #FF9800; }
        
        .metric-highlight {
            background: linear-gradient(135deg, #2c3e50, #4a6491);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        
        .upload-box {
            border: 2px dashed #00b4d8;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
            background: rgba(0, 180, 216, 0.05);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Initialize Session State
def init_session_state():
    if "forecast_data" not in st.session_state:
        st.session_state.forecast_data = None
    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None
    if "model_performance" not in st.session_state:
        st.session_state.model_performance = {}
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "Random Forest"
    if "forecast_result" not in st.session_state:
        st.session_state.forecast_result = None
    if "data_source" not in st.session_state:
        st.session_state.data_source = "sample"

# Generate Sample Data
def generate_sample_data():
    """Generate realistic sample energy consumption data"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    n_days = len(dates)
    
    # Annual seasonality
    annual_season = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    
    # Weekly seasonality (higher on weekdays)
    weekly_pattern = np.array([1.2, 1.1, 1.0, 1.0, 1.1, 0.8, 0.7] * (n_days // 7 + 1))[:n_days]
    
    # Trend (slight increase over time)
    trend = np.linspace(0, 3, n_days)
    
    # Random noise
    noise = np.random.normal(0, 2, n_days)
    
    # Holidays effect
    holidays = np.random.choice([0, 1], n_days, p=[0.95, 0.05])
    holiday_effect = -5 * holidays
    
    # Combine all components
    consumption = 25 + annual_season + weekly_pattern + trend + noise + holiday_effect
    consumption = np.maximum(consumption, 10)
    
    # Generate temperature
    temperature = 15 + 20 * np.sin(2 * np.pi * np.arange(n_days) / 365) + np.random.normal(0, 5, n_days)
    
    # Price variation
    base_price = 8
    price_variation = 1.5 * np.sin(2 * np.pi * np.arange(n_days) / 90) + np.random.normal(0, 0.3, n_days)
    price_per_kwh = base_price + price_variation
    
    # Calculate cost
    cost = consumption * price_per_kwh
    
    df = pd.DataFrame({
        'Date': dates,
        'Energy_Consumption_kWh': np.round(consumption, 2),
        'Temperature_C': np.round(temperature, 1),
        'Price_Rs_per_kWh': np.round(price_per_kwh, 2),
        'Cost_Rs': np.round(cost, 2),
        'Day_of_Week': dates.weekday,
        'Month': dates.month,
        'Is_Weekend': (dates.weekday >= 5).astype(int),
        'Holiday': holidays
    })
    
    return df

# Feature Engineering
def create_features(df, target_col='Energy_Consumption_kWh'):
    """Create comprehensive features for ML models"""
    df = df.copy()
    
    # Basic time features
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['weekofyear'] = df['Date'].dt.isocalendar().week
    df['quarter'] = df['Date'].dt.quarter
    
    # Cyclical features
    df['sin_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    
    # Lag features
    for lag in [1, 2, 7, 14]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    
    # Difference features
    df['diff_1'] = df[target_col].diff(1)
    df['diff_7'] = df[target_col].diff(7)
    
    # Interaction features
    if 'Temperature_C' in df.columns:
        df['temp_squared'] = df['Temperature_C'] ** 2
    
    # Day type features
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df.dropna()

# ML Models Class
class EnergyForecastModels:
    """Class containing different ML models for energy forecasting"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_importance = {}
    
    def prepare_data(self, df, target_col='Energy_Consumption_kWh'):
        """Prepare data for ML models"""
        # Create features
        df_features = create_features(df, target_col)
        
        # Identify feature columns
        exclude_cols = ['Date', target_col, 'Cost_Rs', 'Price_Rs_per_kWh']
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        X = df_features[feature_cols]
        y = df_features[target_col]
        
        # Train-test split (chronological)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def train_random_forest(self, X_train, X_test, y_train, y_test, n_estimators=200, max_depth=15):
        """Train Random Forest model"""
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Feature importance
        self.feature_importance['Random Forest'] = model.feature_importances_
        
        return model, metrics
    
    def train_xgboost(self, X_train, X_test, y_train, y_test, n_estimators=200, learning_rate=0.05):
        """Train XGBoost model"""
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=8,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        self.feature_importance['XGBoost'] = model.feature_importances_
        
        return model, metrics
    
    def train_lightgbm(self, X_train, X_test, y_train, y_test, n_estimators=200, learning_rate=0.05):
        """Train LightGBM model"""
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            learning_rate=learning_rate,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        self.feature_importance['LightGBM'] = model.feature_importances_
        
        return model, metrics
    
    def train_linear_regression(self, X_train, X_test, y_train, y_test):
        """Train Linear Regression with polynomial features"""
        # Create polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Scale polynomial features
        poly_scaler = StandardScaler()
        X_train_poly_scaled = poly_scaler.fit_transform(X_train_poly)
        X_test_poly_scaled = poly_scaler.transform(X_test_poly)
        
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_poly_scaled, y_train)
        
        y_pred = model.predict(X_test_poly_scaled)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        return model, metrics, poly, poly_scaler
    
    def train_arima(self, y_train, y_test):
        """Train ARIMA model"""
        if not STATSMODELS_AVAILABLE:
            return None, {"error": "Statsmodels not installed"}
        
        try:
            # Fit ARIMA model
            model = ARIMA(y_train, order=(5, 1, 2))
            model_fit = model.fit()
            
            # Forecast
            forecast_steps = len(y_test)
            y_pred = model_fit.forecast(steps=forecast_steps)
            
            # Handle index mismatch
            if len(y_pred) > len(y_test):
                y_pred = y_pred[:len(y_test)]
            elif len(y_pred) < len(y_test):
                y_test = y_test[:len(y_pred)]
            
            metrics = self._calculate_metrics(y_test, y_pred)
            
            return model_fit, metrics
        except Exception as e:
            return None, {"error": str(e)}
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100,
            'Max_Error': np.max(np.abs(y_true - y_pred))
        }

# Forecasting with ML Models
def generate_ml_forecast(model_info, last_data, periods=365, target_col='Energy_Consumption_kWh'):
    """Generate forecast using trained ML model"""
    model_type = model_info.get('model_type', 'Random Forest')
    model = model_info.get('model')
    scaler = model_info.get('scaler')
    feature_cols = model_info.get('feature_cols', [])
    
    if model is None:
        return None
    
    forecasts = []
    lower_bounds = []
    upper_bounds = []
    
    # Prepare initial data
    current_data = last_data.copy()
    
    for i in range(periods):
        # Create next date
        next_date = current_data['Date'].max() + pd.Timedelta(days=1)
        
        # Create features for next day
        temp_df = pd.DataFrame({
            'Date': [next_date],
            target_col: [np.nan]  # Will be predicted
        })
        
        # Add temperature if available (use seasonal average)
        if 'Temperature_C' in current_data.columns:
            day_of_year = next_date.dayofyear
            seasonal_temp = 15 + 20 * np.sin(2 * np.pi * day_of_year / 365)
            temp_df['Temperature_C'] = [seasonal_temp]
        
        # Combine with current data
        temp_combined = pd.concat([current_data, temp_df], ignore_index=True)
        df_features = create_features(temp_combined, target_col)
        
        # Get last row's features
        if len(df_features) > 0:
            last_features = df_features.iloc[-1:][feature_cols].fillna(method='ffill').fillna(0)
            
            # Scale features
            if scaler:
                X_pred_scaled = scaler.transform(last_features)
            else:
                X_pred_scaled = last_features
            
            # Make prediction
            if model_type == 'ARIMA' and STATSMODELS_AVAILABLE:
                prediction = model.forecast(steps=1)[0]
            else:
                prediction = model.predict(X_pred_scaled)[0]
            
            # Add uncertainty
            uncertainty = prediction * 0.12  # 12% uncertainty
            forecasts.append(max(0, prediction))
            lower_bounds.append(max(0, prediction - uncertainty))
            upper_bounds.append(max(0, prediction + uncertainty))
            
            # Update current data with prediction
            temp_df[target_col] = prediction
            current_data = pd.concat([current_data, temp_df], ignore_index=True)
    
    # Create forecast dataframe
    forecast_dates = pd.date_range(
        start=last_data['Date'].max() + pd.Timedelta(days=1),
        periods=periods,
        freq='D'
    )
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast_kWh': np.round(forecasts, 2),
        'Lower_Bound': np.round(lower_bounds, 2),
        'Upper_Bound': np.round(upper_bounds, 2),
        'Forecast_Cost_Rs': np.round(np.array(forecasts) * 8, 2)
    })
    
    return forecast_df

# Data Upload and Parsing
def create_data_sidebar():
    """Create sidebar for data upload and parsing"""
    with st.sidebar:
        st.markdown('<h2 style="color: #00b4d8;">📁 Data Source</h2>', unsafe_allow_html=True)
        
        # Data Source Selection
        data_source = st.radio(
            "Choose Data Source:",
            ["📊 Use Sample Data", "📁 Upload Your Data"],
            key="data_source_radio"
        )
        
        st.session_state.data_source = "sample" if data_source == "📊 Use Sample Data" else "upload"
        
        if st.session_state.data_source == "upload":
            st.divider()
            st.markdown("### 📤 Upload Your Data")
            
            uploaded_file = st.file_uploader(
                "Upload Excel/CSV file",
                type=['xlsx', 'xls', 'csv'],
                help="Upload your energy consumption data"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ File uploaded: {uploaded_file.name}")
                    
                    # Show preview
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head(), use_container_width=True)
                    
                    # Auto-detect columns
                    st.markdown("### 🗂️ Column Mapping")
                    
                    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    # Date column selection
                    date_col = st.selectbox(
                        "Select Date Column",
                        options=['Auto-detect'] + df.columns.tolist(),
                        index=0 if not date_columns else 1
                    )
                    
                    if date_col == 'Auto-detect' and date_columns:
                        date_col = date_columns[0]
                    elif date_col == 'Auto-detect':
                        st.warning("No date column detected. Please select manually.")
                        return None
                    
                    # Convert to datetime
                    df[date_col] = pd.to_datetime(df[date_col])
                    
                    # Consumption column selection
                    consumption_col = st.selectbox(
                        "Select Energy Consumption Column",
                        options=['Auto-detect'] + numeric_columns,
                        index=0
                    )
                    
                    if consumption_col == 'Auto-detect' and numeric_columns:
                        consumption_col = numeric_columns[0]
                    
                    # Rename columns to standard names
                    df_clean = df.rename(columns={
                        date_col: 'Date',
                        consumption_col: 'Energy_Consumption_kWh'
                    })
                    
                    # Select additional columns
                    st.markdown("#### Additional Data (Optional)")
                    
                    temp_cols = [col for col in df.columns if 'temp' in col.lower() or 'degree' in col.lower()]
                    cost_cols = [col for col in df.columns if 'cost' in col.lower() or 'price' in col.lower()]
                    
                    if temp_cols:
                        temp_col = st.selectbox(
                            "Temperature Column",
                            ['None'] + temp_cols,
                            index=0
                        )
                        if temp_col != 'None':
                            df_clean['Temperature_C'] = df[temp_col]
                    
                    if cost_cols:
                        cost_col = st.selectbox(
                            "Cost Column",
                            ['None'] + cost_cols,
                            index=0
                        )
                        if cost_col != 'None':
                            df_clean['Cost_Rs'] = df[cost_col]
                    
                    st.session_state.forecast_data = df_clean
                    st.success("✅ Data parsed successfully!")
                    
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    return None
        
        else:
            # Sample data option
            st.divider()
            if st.button("🔄 Generate Sample Data", use_container_width=True):
                sample_data = generate_sample_data()
                st.session_state.forecast_data = sample_data
                st.success("✅ Sample data generated!")
                st.rerun()
        
        return True

# ML Model Selection Sidebar
def create_ml_sidebar():
    """Create sidebar for ML model selection"""
    with st.sidebar:
        st.markdown("---")
        st.markdown('<h2 style="color: #00b4d8;">🤖 ML Configuration</h2>', unsafe_allow_html=True)
        
        # Model Selection
        model_options = ["Random Forest", "XGBoost", "LightGBM", "Linear Regression"]
        if STATSMODELS_AVAILABLE:
            model_options.append("ARIMA")
        
        selected_model = st.selectbox(
            "Select ML Model:",
            model_options,
            index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
        )
        
        st.session_state.selected_model = selected_model
        
        # Model-specific parameters
        st.divider()
        st.markdown("### ⚙️ Model Parameters")
        
        if selected_model == "Random Forest":
            n_estimators = st.slider("Number of Trees", 50, 500, 200, 50)
            max_depth = st.slider("Max Depth", 5, 30, 15, 1)
            st.session_state.model_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth
            }
        
        elif selected_model == "XGBoost":
            n_estimators = st.slider("Number of Trees", 50, 500, 200, 50)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05, 0.01)
            st.session_state.model_params = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate
            }
        
        elif selected_model == "LightGBM":
            n_estimators = st.slider("Number of Trees", 50, 500, 200, 50)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05, 0.01)
            st.session_state.model_params = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate
            }
        
        # Forecast Settings
        st.divider()
        st.markdown("### 📅 Forecast Settings")
        
        forecast_period = st.selectbox(
            "Forecast Period:",
            ["Next 30 days", "Next 90 days", "Next 180 days", "Next 365 days"],
            index=3
        )
        
        period_map = {
            "Next 30 days": 30,
            "Next 90 days": 90,
            "Next 180 days": 180,
            "Next 365 days": 365
        }
        
        st.session_state.forecast_period = period_map[forecast_period]
        
        # Training Settings
        st.divider()
        train_split = st.slider(
            "Training Data %",
            min_value=60,
            max_value=90,
            value=80,
            help="Percentage of data used for training"
        )
        
        st.session_state.train_split = train_split
        
        # Action Button
        st.divider()
        if st.button("🚀 Train & Forecast", type="primary", use_container_width=True):
            return True
        
        return False

# Main Forecast Page
def main():
    load_css()
    init_session_state()
    
    # Title
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown('<h1 style="color: #00b4d8;">🤖 ML Energy Forecast</h1>', unsafe_allow_html=True)
        st.markdown("### Advanced Machine Learning for Energy Consumption Prediction")
    
    with col2:
        st.info(f"Using: {st.session_state.data_source.title()} Data")
    
    st.divider()
    
    # Data Upload Section
    if st.session_state.forecast_data is None:
        st.markdown("### 📊 Load Your Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="upload-box">
                <h3>📊 Sample Data</h3>
                <p>Use pre-generated sample data with realistic patterns</p>
                <p><small>1 year of simulated energy data with seasonality</small></p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Use Sample Data", key="use_sample", use_container_width=True):
                sample_data = generate_sample_data()
                st.session_state.forecast_data = sample_data
                st.session_state.data_source = "sample"
                st.rerun()
        
        with col2:
            st.markdown("""
            <div class="upload-box">
                <h3>📁 Upload Your Data</h3>
                <p>Upload your own energy consumption data</p>
                <p><small>Supported: Excel (.xlsx, .xls), CSV (.csv)</small></p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Upload Data", key="upload_data", use_container_width=True):
                st.session_state.data_source = "upload"
                st.rerun()
        
        # Create data sidebar
        create_data_sidebar()
        return
    
    data = st.session_state.forecast_data
    
    # Data Overview
    st.subheader("📋 Data Overview")
    
    # Determine target column
    target_col = 'Energy_Consumption_kWh' if 'Energy_Consumption_kWh' in data.columns else 'Consumption'
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>{len(data):,}</h3>
            <p>Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        date_range = f"{data['Date'].min().strftime('%b %d, %Y')} to {data['Date'].max().strftime('%b %d, %Y')}"
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>{date_range}</h3>
            <p>Date Range</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_cons = data[target_col].mean()
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>{avg_cons:.1f} kWh</h3>
            <p>Avg. Daily</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if 'Cost_Rs' in data.columns:
            avg_cost = data['Cost_Rs'].mean()
            st.markdown(f"""
            <div class="metric-highlight">
                <h3>₹{avg_cost:.0f}</h3>
                <p>Avg. Daily Cost</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            est_cost = avg_cons * 8
            st.markdown(f"""
            <div class="metric-highlight">
                <h3>₹{est_cost:.0f}</h3>
                <p>Est. Daily Cost</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Data Visualization
    st.subheader("📈 Data Visualization")
    
    tab1, tab2 = st.tabs(["📊 Daily Pattern", "📅 Monthly Summary"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data[target_col],
            mode='lines',
            name='Consumption',
            line=dict(color='#00b4d8', width=2)
        ))
        
        # Add 7-day moving average
        if len(data) > 7:
            data_sorted = data.sort_values('Date')
            data_sorted['Moving_Avg'] = data_sorted[target_col].rolling(window=7).mean()
            fig.add_trace(go.Scatter(
                x=data_sorted['Date'],
                y=data_sorted['Moving_Avg'],
                mode='lines',
                name='7-day Moving Avg',
                line=dict(color='#ff6b6b', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title="Daily Energy Consumption",
            xaxis_title="Date",
            yaxis_title="Consumption (kWh)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        data['Month'] = data['Date'].dt.to_period('M')
        monthly_data = data.groupby('Month').agg({
            target_col: ['sum', 'mean', 'std']
        }).round(2)
        monthly_data.columns = ['Total', 'Average', 'Std Dev']
        monthly_data = monthly_data.reset_index()
        monthly_data['Month'] = monthly_data['Month'].astype(str)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_data['Month'],
            y=monthly_data['Total'],
            name='Total Consumption',
            marker_color='#4ecdc4'
        ))
        
        fig.update_layout(
            title="Monthly Energy Consumption",
            xaxis_title="Month",
            yaxis_title="Total Consumption (kWh)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Create sidebars
    create_data_sidebar()
    should_train = create_ml_sidebar()
    
    # Training Section
    if should_train and st.session_state.forecast_data is not None:
        with st.spinner(f"Training {st.session_state.selected_model} model..."):
            # Initialize model trainer
            trainer = EnergyForecastModels()
            
            # Prepare data
            X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(data, target_col)
            
            # Get model parameters
            params = st.session_state.get('model_params', {})
            
            # Train selected model
            if st.session_state.selected_model == "Random Forest":
                n_estimators = params.get('n_estimators', 200)
                max_depth = params.get('max_depth', 15)
                model, metrics = trainer.train_random_forest(X_train, X_test, y_train, y_test, n_estimators, max_depth)
                model_type = 'Random Forest'
            
            elif st.session_state.selected_model == "XGBoost":
                n_estimators = params.get('n_estimators', 200)
                learning_rate = params.get('learning_rate', 0.05)
                model, metrics = trainer.train_xgboost(X_train, X_test, y_train, y_test, n_estimators, learning_rate)
                model_type = 'XGBoost'
            
            elif st.session_state.selected_model == "LightGBM":
                n_estimators = params.get('n_estimators', 200)
                learning_rate = params.get('learning_rate', 0.05)
                model, metrics = trainer.train_lightgbm(X_train, X_test, y_train, y_test, n_estimators, learning_rate)
                model_type = 'LightGBM'
            
            elif st.session_state.selected_model == "Linear Regression":
                model, metrics, poly, poly_scaler = trainer.train_linear_regression(X_train, X_test, y_train, y_test)
                model_type = 'Linear Regression'
                model_info = {
                    'model': model,
                    'scaler': poly_scaler,
                    'feature_cols': feature_cols,
                    'model_type': model_type,
                    'poly': poly
                }
            elif st.session_state.selected_model == "ARIMA":
                model, metrics = trainer.train_arima(y_train, y_test)
                model_type = 'ARIMA'
                model_info = {
                    'model': model,
                    'scaler': None,
                    'feature_cols': [],
                    'model_type': model_type
                }
            
            # Store model info for non-ARIMA models
            if st.session_state.selected_model != "ARIMA" and st.session_state.selected_model != "Linear Regression":
                model_info = {
                    'model': model,
                    'scaler': trainer.scaler,
                    'feature_cols': feature_cols,
                    'model_type': model_type
                }
            
            # Store results
            st.session_state.trained_model = model_info
            st.session_state.model_performance = metrics
            
            st.success(f"✅ {st.session_state.selected_model} model trained successfully!")
            
            # Display performance
            st.subheader("🎯 Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MAE", f"{metrics.get('MAE', 0):.2f} kWh", 
                         delta="Lower is better", delta_color="inverse")
            
            with col2:
                st.metric("RMSE", f"{metrics.get('RMSE', 0):.2f} kWh",
                         delta="Lower is better", delta_color="inverse")
            
            with col3:
                r2 = metrics.get('R2', 0)
                status = "Excellent" if r2 > 0.8 else "Good" if r2 > 0.6 else "Fair" if r2 > 0.4 else "Poor"
                st.metric("R² Score", f"{r2:.3f}", delta=status)
            
            with col4:
                mape = metrics.get('MAPE', 0)
                status = "Excellent" if mape < 5 else "Good" if mape < 10 else "Fair" if mape < 15 else "Poor"
                st.metric("MAPE", f"{mape:.1f}%", delta=status, delta_color="inverse")
            
            # Feature Importance for tree-based models
            if model_type in ['Random Forest', 'XGBoost', 'LightGBM']:
                if hasattr(trainer, 'feature_importance') and model_type in trainer.feature_importance:
                    st.subheader("🔍 Top 10 Feature Importance")
                    
                    importance_scores = trainer.feature_importance[model_type]
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': importance_scores
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig = go.Figure(go.Bar(
                        x=feature_importance_df['Importance'],
                        y=feature_importance_df['Feature'],
                        orientation='h',
                        marker=dict(color='#00b4d8')
                    ))
                    
                    fig.update_layout(
                        title=f"Feature Importance - {model_type}",
                        xaxis_title="Importance Score",
                        yaxis_title="Feature",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Generate forecast
            with st.spinner("Generating forecast..."):
                forecast_df = generate_ml_forecast(
                    model_info,
                    data,
                    st.session_state.forecast_period,
                    target_col
                )
                
                if forecast_df is not None:
                    st.session_state.forecast_result = forecast_df
                    st.success("✅ Forecast generated successfully!")
                else:
                    st.error("Failed to generate forecast")
    
    # Display Forecast Results
    if "forecast_result" in st.session_state and st.session_state.forecast_result is not None:
        forecast_data = st.session_state.forecast_result
        
        st.divider()
        st.subheader("📈 Forecast Results")
        
        # Forecast Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_forecast = forecast_data['Forecast_kWh'].sum()
            st.markdown(f"""
            <div class="ml-card rf-card">
                <h3>{total_forecast:,.0f} kWh</h3>
                <p>Total Forecast</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_daily = forecast_data['Forecast_kWh'].mean()
            st.markdown(f"""
            <div class="ml-card xgb-card">
                <h3>{avg_daily:.1f} kWh/day</h3>
                <p>Avg Daily</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_cost = forecast_data['Forecast_Cost_Rs'].sum()
            st.markdown(f"""
            <div class="ml-card lgb-card">
                <h3>₹{total_cost:,.0f}</h3>
                <p>Total Cost</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            historical_avg = data[target_col].mean()
            saving_pct = ((historical_avg - avg_daily) / historical_avg) * 100
            st.markdown(f"""
            <div class="ml-card lr-card">
                <h3>{saving_pct:.1f}%</h3>
                <p>Est. Savings</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Forecast Visualization
        st.subheader("📊 Forecast Visualization")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Energy Consumption Forecast", "Cost Forecast"),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        # Energy Forecast with confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_data['Date'],
            y=forecast_data['Upper_Bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=forecast_data['Date'],
            y=forecast_data['Lower_Bound'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0, 180, 216, 0.2)',
            fill='tonexty',
            name='90% Confidence Interval'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=forecast_data['Date'],
            y=forecast_data['Forecast_kWh'],
            mode='lines',
            name='Forecast',
            line=dict(color='#00b4d8', width=3)
        ), row=1, col=1)
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data[target_col],
            mode='lines',
            name='Historical',
            line=dict(color='#FF6B6B', width=2)
        ), row=1, col=1)
        
        # Cost Forecast
        fig.add_trace(go.Scatter(
            x=forecast_data['Date'],
            y=forecast_data['Forecast_Cost_Rs'],
            mode='lines',
            name='Cost Forecast',
            line=dict(color='#4ecdc4', width=3)
        ), row=2, col=1)
        
        fig.update_layout(
            height=700,
            template="plotly_dark",
            showlegend=True,
            title=f"Forecast using {st.session_state.selected_model} Model"
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="kWh", row=1, col=1)
        fig.update_yaxes(title_text="₹", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast Details
        with st.expander("📋 Detailed Forecast Table"):
            display_cols = ['Date', 'Forecast_kWh', 'Lower_Bound', 'Upper_Bound', 'Forecast_Cost_Rs']
            display_df = forecast_data[display_cols].copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_df.head(30), use_container_width=True)
            
            if len(display_df) > 30:
                st.caption(f"Showing first 30 of {len(display_df)} records")
            
            # Download option
            csv = forecast_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast as CSV",
                data=csv,
                file_name=f"energy_forecast_{st.session_state.selected_model.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Action Buttons
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Train New Model", use_container_width=True):
            st.session_state.trained_model = None
            st.session_state.forecast_result = None
            st.rerun()
    
    with col2:
        if st.button("📁 Load Different Data", use_container_width=True):
            st.session_state.forecast_data = None
            st.session_state.trained_model = None
            st.session_state.forecast_result = None
            st.rerun()
    
    with col3:
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            st.switch_page("main.py")

if __name__ == "__main__":
    main()
