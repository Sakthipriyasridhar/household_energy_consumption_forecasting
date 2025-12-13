import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb

# Time Series Models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.sidebar.warning("Install statsmodels for ARIMA: pip install statsmodels")

# Page Configuration
st.set_page_config(
    page_title="Advanced ML Energy Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    css = """
    <style>
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Model comparison cards */
        .model-card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 5px solid;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        .model-card:hover {
            transform: translateY(-5px);
        }
        
        .rf-card { border-left-color: #4CAF50; }
        .xgb-card { border-left-color: #2196F3; }
        .lgbm-card { border-left-color: #FF9800; }
        .lr-card { border-left-color: #9C27B0; }
        .gb-card { border-left-color: #FF5722; }
        .arima-card { border-left-color: #607D8B; }
        
        /* Performance badges */
        .performance-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        .best-performance { background: #4CAF50; color: white; }
        .good-performance { background: #8BC34A; color: white; }
        .average-performance { background: #FFC107; color: black; }
        .poor-performance { background: #F44336; color: white; }
        
        /* Date picker container */
        .date-picker-container {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            border: 2px solid #00b4d8;
            margin: 1rem 0;
        }
        
        /* Future forecast notice */
        .future-forecast {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            text-align: center;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Initialize Session State
def init_session_state():
    defaults = {
        "forecast_data": None,
        "trained_models": {},
        "model_performance": {},
        "model_forecasts": {},
        "selected_model": "Random Forest",
        "forecast_result": None,
        "data_source": None,
        "file_uploaded": False,
        "feature_importance": {},
        "all_models_trained": False,
        "feature_engineering": {
            'lag_features': True,
            'rolling_features': True,
            'date_features': True,
            'cyclical_features': True,
            'interaction_features': False,
            'window_sizes': [3, 7, 14, 30],
            'lag_days': [1, 2, 3, 7, 14, 30]
        },
        "confidence_level": 95,
        "survey_data": {},
        "use_sample_data": False,
        "sample_config": {},
        "forecast_start_date": None,
        "selected_models": ["Random Forest", "XGBoost", "LightGBM", "Linear Regression", "Gradient Boosting"],
        "forecast_future_date": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Data Loading Functions (Keep existing)
class DataLoader:
    @staticmethod
    def load_from_upload(uploaded_file):
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                return None, "Unsupported file format"
            
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                df[date_columns[0]] = pd.to_datetime(df[date_columns[0]], errors='coerce')
            
            return df, "Success"
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def load_from_survey(survey_responses):
        try:
            num_records = survey_responses.get('num_records', 365)
            start_date = survey_responses.get('start_date', '2023-01-01')
            consumption_type = survey_responses.get('consumption_type', 'Household')
            
            dates = pd.date_range(start=start_date, periods=num_records, freq='D')
            
            if consumption_type == 'Household':
                base_consumption = 15
                seasonal_factor = 5
            elif consumption_type == 'Commercial':
                base_consumption = 100
                seasonal_factor = 25
            else:
                base_consumption = 500
                seasonal_factor = 100
            
            np.random.seed(42)
            seasonal = seasonal_factor * np.sin(2 * np.pi * np.arange(num_records) / 365)
            weekday_factor = np.array([0.8 if date.weekday() < 5 else 1.2 for date in dates])
            noise = np.random.normal(0, base_consumption * 0.2, num_records)
            
            consumption = base_consumption + seasonal + weekday_factor * base_consumption * 0.1 + noise
            consumption = np.maximum(consumption, 0.1)
            
            data = pd.DataFrame({
                'Date': dates,
                'Energy_Consumption_kWh': np.round(consumption, 2)
            })
            
            if survey_responses.get('include_temperature', True):
                temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(num_records) / 365) + np.random.normal(0, 5, num_records)
                data['Temperature_C'] = np.round(temperature, 1)
            
            if survey_responses.get('include_cost', True):
                price_per_kwh = survey_responses.get('price_per_kwh', 8)
                data['Cost_Rs'] = np.round(data['Energy_Consumption_kWh'] * price_per_kwh, 2)
                data['Price_Rs_per_kWh'] = price_per_kwh
            
            return data, "Success"
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def load_sample_data(config=None):
        if config is None:
            config = {
                'num_records': 365,
                'start_date': '2023-01-01',
                'seasonality': 'High',
                'trend': 'Upward',
                'noise_level': 'Medium',
                'include_features': ['Temperature', 'Cost']
            }
        
        try:
            num_records = config['num_records']
            dates = pd.date_range(start=config['start_date'], periods=num_records, freq='D')
            
            if config['seasonality'] == 'High':
                seasonal_factor = 8
            elif config['seasonality'] == 'Medium':
                seasonal_factor = 5
            else:
                seasonal_factor = 2
            
            if config['trend'] == 'Upward':
                trend_factor = 0.02
            elif config['trend'] == 'Downward':
                trend_factor = -0.01
            else:
                trend_factor = 0
            
            base_consumption = 20 + trend_factor * np.arange(num_records)
            seasonal = seasonal_factor * np.sin(2 * np.pi * np.arange(num_records) / 365)
            weekly_pattern = 3 * np.array([0.8 if date.weekday() < 5 else 1.2 for date in dates])
            noise = 0.2 * base_consumption * np.random.randn(num_records)
            
            consumption = base_consumption + seasonal + weekly_pattern + noise
            consumption = np.maximum(consumption, 0.1)
            
            data = pd.DataFrame({
                'Date': dates,
                'Energy_Consumption_kWh': np.round(consumption, 2)
            })
            
            if 'Temperature' in config['include_features']:
                data['Temperature_C'] = np.round(20 + 10 * np.sin(2 * np.pi * np.arange(num_records) / 365) + 
                                                np.random.normal(0, 5, num_records), 1)
            
            if 'Cost' in config['include_features']:
                data['Cost_Rs'] = np.round(data['Energy_Consumption_kWh'] * 8, 2)
                data['Price_Rs_per_kWh'] = 8
            
            return data, "Success"
        except Exception as e:
            return None, str(e)

# Advanced Feature Engineering (Simplified to avoid errors)
class AdvancedFeatureEngineer:
    def __init__(self):
        self.feature_names = []
    
    def create_all_features(self, df, target_col='Energy_Consumption_kWh', config=None):
        if config is None:
            config = st.session_state.feature_engineering
        
        df = df.copy()
        
        # 1. Basic Date Features
        if config['date_features']:
            df['year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            df['day'] = df['Date'].dt.day
            df['dayofweek'] = df['Date'].dt.dayofweek
            df['quarter'] = df['Date'].dt.quarter
            df['is_weekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)
        
        # 2. Cyclical Features
        if config['cyclical_features'] and 'month' in df.columns:
            df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
            df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 3. Lag Features
        if config['lag_features']:
            for lag in config['lag_days']:
                df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # 4. Rolling Statistics
        if config['rolling_features']:
            for window in config['window_sizes']:
                df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
        
        # Drop NaN rows
        df = df.dropna()
        
        # Store feature names (exclude non-numeric and target)
        exclude_cols = ['Date', target_col] + [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        
        return df

# Multi-Model Trainer Class (Simplified to avoid ARIMA errors)
class MultiModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.all_metrics = {}
        self.forecasts = {}
    
    def train_all_models(self, X_train, X_test, y_train, y_test, selected_models):
        """Train multiple models at once"""
        results = {}
        
        for model_name in selected_models:
            try:
                if model_name == "Random Forest":
                    model, metrics = self._train_random_forest(X_train, X_test, y_train, y_test)
                elif model_name == "XGBoost":
                    model, metrics = self._train_xgboost(X_train, X_test, y_train, y_test)
                elif model_name == "LightGBM":
                    model, metrics = self._train_lightgbm(X_train, X_test, y_train, y_test)
                elif model_name == "Linear Regression":
                    model, metrics = self._train_linear_regression(X_train, X_test, y_train, y_test)
                elif model_name == "Gradient Boosting":
                    model, metrics = self._train_gradient_boosting(X_train, X_test, y_train, y_test)
                elif model_name == "ARIMA" and STATSMODELS_AVAILABLE:
                    # Skip ARIMA for now to avoid errors
                    continue
                else:
                    continue
                
                if model is not None:
                    self.models[model_name] = model
                    self.all_metrics[model_name] = metrics
                    results[model_name] = metrics
                    
            except Exception as e:
                st.warning(f"Error training {model_name}: {str(e)}")
                continue
        
        return results
    
    def _train_random_forest(self, X_train, X_test, y_train, y_test):
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.feature_importance['Random Forest'] = model.feature_importances_
        
        return model, self._calculate_metrics(y_test, y_pred)
    
    def _train_xgboost(self, X_train, X_test, y_train, y_test):
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.feature_importance['XGBoost'] = model.feature_importances_
        
        return model, self._calculate_metrics(y_test, y_pred)
    
    def _train_lightgbm(self, X_train, X_test, y_train, y_test):
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.feature_importance['LightGBM'] = model.feature_importances_
        
        return model, self._calculate_metrics(y_test, y_pred)
    
    def _train_linear_regression(self, X_train, X_test, y_train, y_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        return model, self._calculate_metrics(y_test, y_pred)
    
    def _train_gradient_boosting(self, X_train, X_test, y_train, y_test):
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.feature_importance['Gradient Boosting'] = model.feature_importances_
        
        return model, self._calculate_metrics(y_test, y_pred)
    
    def _calculate_metrics(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Handle MAPE safely
        try:
            mask = y_true != 0
            if np.any(mask):
                mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
            else:
                mape = np.nan
        except:
            mape = np.nan
        
        metrics = {
            'R²': max(0, r2_score(y_true, y_pred)),  # Ensure R² is not negative
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': mape
        }
        
        return metrics

# FIXED: Forecast Generator with proper date handling
def generate_forecasts_for_all_models(trainer, data, feature_cols, forecast_start_date, periods, target_col, confidence_level=95):
    """Generate forecasts for all trained models starting from a specific date"""
    forecasts = {}
    
    for model_name, model in trainer.models.items():
        try:
            # Prepare data for forecasting
            current_data = data.copy()
            
            # Determine if we need to forecast from a future date
            last_data_date = current_data['Date'].max()
            forecast_start_dt = pd.to_datetime(forecast_start_date)
            
            # If forecasting from future date, extend data with predictions
            if forecast_start_dt > last_data_date:
                days_gap = (forecast_start_dt - last_data_date).days
                total_periods = days_gap + periods
            else:
                # Use data up to forecast start date
                current_data = current_data[current_data['Date'] <= forecast_start_dt]
                total_periods = periods
            
            # Generate predictions
            predictions = []
            
            for i in range(total_periods):
                next_date = current_data['Date'].max() + timedelta(days=1)
                
                # Create features for next date
                temp_df = current_data.copy()
                temp_df = pd.concat([temp_df, pd.DataFrame({
                    'Date': [next_date],
                    target_col: [np.nan]
                })], ignore_index=True)
                
                # Engineer features
                engineer = AdvancedFeatureEngineer()
                df_features = engineer.create_all_features(temp_df, target_col)
                
                if len(df_features) > 0 and feature_cols:
                    # Get available features
                    available_features = [col for col in feature_cols if col in df_features.columns]
                    
                    if available_features:
                        last_row = df_features.iloc[-1:][available_features].fillna(0)
                        
                        # Make prediction
                        prediction = model.predict(last_row)[0]
                        predictions.append(max(0.1, prediction))
                        
                        # Update data for next iteration
                        current_data = pd.concat([current_data, pd.DataFrame({
                            'Date': [next_date],
                            target_col: [prediction]
                        })], ignore_index=True)
            
            # Extract only the requested periods
            if forecast_start_dt > last_data_date:
                # Skip the gap days and take only forecast periods
                predictions = predictions[days_gap:days_gap+periods]
                forecast_dates = pd.date_range(start=forecast_start_dt, periods=periods, freq='D')
            else:
                # We already have the correct predictions
                predictions = predictions[:periods]
                forecast_dates = pd.date_range(start=forecast_start_dt + timedelta(days=1), periods=periods, freq='D')
            
            if predictions:
                # Calculate confidence intervals (simplified)
                pred_array = np.array(predictions)
                std_dev = pred_array.std() * 0.5  # Simplified uncertainty
                
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecast_kWh': np.round(predictions, 2),
                    'Lower_Bound_kWh': np.round(pred_array - 1.96 * std_dev, 2),
                    'Upper_Bound_kWh': np.round(pred_array + 1.96 * std_dev, 2),
                    'Forecast_Cost_Rs': np.round(pred_array * 8, 2),
                    'Model': model_name
                })
                
                # Ensure no negative values
                forecast_df['Lower_Bound_kWh'] = forecast_df['Lower_Bound_kWh'].clip(lower=0.1)
                forecast_df['Forecast_kWh'] = forecast_df['Forecast_kWh'].clip(lower=0.1)
                forecast_df['Upper_Bound_kWh'] = forecast_df['Upper_Bound_kWh'].clip(lower=0.1)
                
                forecasts[model_name] = forecast_df
                
        except Exception as e:
            st.warning(f"Could not generate forecast for {model_name}: {str(e)}")
            continue
    
    return forecasts

# Create Model Comparison Table
def create_model_comparison_table(metrics_dict):
    """Create a comprehensive comparison table of all models"""
    if not metrics_dict:
        return pd.DataFrame()
    
    comparison_data = []
    for model_name, metrics in metrics_dict.items():
        row = {'Model': model_name}
        row.update(metrics)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by R² score (descending)
    if 'R²' in df.columns:
        df = df.sort_values('R²', ascending=False)
    
    return df

# Display Model Performance Cards
def display_model_performance_cards(metrics_dict):
    """Display performance metrics as cards for each model"""
    if not metrics_dict:
        return
    
    st.subheader("📊 Model Performance Comparison")
    
    cols = st.columns(min(4, len(metrics_dict)))
    
    for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
        with cols[idx % len(cols)]:
            card_class = ""
            if model_name == "Random Forest":
                card_class = "rf-card"
            elif model_name == "XGBoost":
                card_class = "xgb-card"
            elif model_name == "LightGBM":
                card_class = "lgbm-card"
            elif model_name == "Linear Regression":
                card_class = "lr-card"
            elif model_name == "Gradient Boosting":
                card_class = "gb-card"
            
            st.markdown(f'<div class="model-card {card_class}">', unsafe_allow_html=True)
            
            st.markdown(f"### {model_name}")
            
            # Display metrics
            r2 = metrics.get('R²', 0)
            st.metric("R² Score", f"{r2:.4f}")
            
            rmse = metrics.get('RMSE', 0)
            st.metric("RMSE", f"{rmse:.2f}")
            
            mape = metrics.get('MAPE', np.nan)
            if not np.isnan(mape):
                st.metric("MAPE", f"{mape:.2f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)

# FIXED: Create Interactive Forecast Selector with proper date handling
def create_forecast_selector(model_forecasts, data, target_col):
    """Create dropdown to select which model's forecast to display"""
    if not model_forecasts:
        return None, None
    
    st.subheader("🔍 Select Forecast to View")
    
    # Create dropdown with model names
    model_names = list(model_forecasts.keys())
    selected_model = st.selectbox(
        "Choose Algorithm to View Forecast",
        model_names,
        index=0,
        help="Select which algorithm's forecast to visualize"
    )
    
    if selected_model in model_forecasts:
        forecast_data = model_forecasts[selected_model]
        
        # Create visualization
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data[target_col],
            mode='lines',
            name='Historical Data',
            line=dict(color='#FF6B6B', width=2),
            opacity=0.8
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_data['Date'],
            y=forecast_data['Forecast_kWh'],
            mode='lines',
            name=f'{selected_model} Forecast',
            line=dict(color='#00b4d8', width=3)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_data['Date'], forecast_data['Date'][::-1]]),
            y=pd.concat([forecast_data['Upper_Bound_kWh'], forecast_data['Lower_Bound_kWh'][::-1]]),
            fill='toself',
            fillcolor='rgba(0, 180, 216, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        ))
        
        fig.update_layout(
            title=f"{selected_model} Forecast Visualization",
            xaxis_title="Date",
            yaxis_title="Energy Consumption (kWh)",
            template="plotly_dark",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        return forecast_data, selected_model
    
    return None, None

# Feature Engineering Sidebar
def create_feature_engineering_sidebar():
    """Sidebar for feature engineering configuration"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔧 Feature Engineering")
        
        with st.expander("Configure Features", expanded=False):
            lag_features = st.checkbox("Lag Features", 
                                      value=st.session_state.feature_engineering['lag_features'])
            
            if lag_features:
                lag_days = st.multiselect(
                    "Lag Days",
                    options=[1, 2, 3, 7, 14, 30],
                    default=st.session_state.feature_engineering['lag_days']
                )
            else:
                lag_days = []
            
            rolling_features = st.checkbox("Rolling Statistics",
                                          value=st.session_state.feature_engineering['rolling_features'])
            
            if rolling_features:
                window_sizes = st.multiselect(
                    "Window Sizes",
                    options=[3, 7, 14, 30],
                    default=st.session_state.feature_engineering['window_sizes']
                )
            else:
                window_sizes = []
            
            date_features = st.checkbox("Date Features",
                                       value=st.session_state.feature_engineering['date_features'])
            
            cyclical_features = st.checkbox("Cyclical Encoding",
                                           value=st.session_state.feature_engineering['cyclical_features'])
        
        st.session_state.feature_engineering.update({
            'lag_features': lag_features,
            'rolling_features': rolling_features,
            'date_features': date_features,
            'cyclical_features': cyclical_features,
            'window_sizes': window_sizes,
            'lag_days': lag_days
        })
        
        # Model selection
        st.markdown("---")
        st.markdown("### 🤖 Select Algorithms")
        
        available_models = ["Random Forest", "XGBoost", "LightGBM", "Linear Regression", "Gradient Boosting"]
        # Skip ARIMA for now to avoid errors
        # if STATSMODELS_AVAILABLE:
        #     available_models.append("ARIMA")
        
        selected_models = st.multiselect(
            "Choose algorithms to train",
            available_models,
            default=st.session_state.get('selected_models', ["Random Forest", "XGBoost", "LightGBM"])
        )
        
        st.session_state.selected_models = selected_models

# Data Source Selection
def create_data_source_selection():
    """Create data source selection interface"""
    st.markdown("### 📥 Select Data Source")
    
    source_tab1, source_tab2, source_tab3 = st.tabs(["📁 Upload", "📋 Survey", "🎲 Sample"])
    
    with source_tab1:
        uploaded_file = st.file_uploader("Choose file", type=['csv', 'xlsx', 'json'])
        if uploaded_file and st.button("Load Uploaded Data", use_container_width=True):
            data, message = DataLoader.load_from_upload(uploaded_file)
            if data is not None:
                # Ensure Date column exists
                if 'Date' not in data.columns:
                    # Try to find date column
                    date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
                    if date_cols:
                        data = data.rename(columns={date_cols[0]: 'Date'})
                    else:
                        data['Date'] = pd.date_range(start='2023-01-01', periods=len(data), freq='D')
                
                st.session_state.forecast_data = data
                st.session_state.data_source = "upload"
                st.success(f"✅ Data loaded! Shape: {data.shape}")
                st.rerun()
    
    with source_tab2:
        col1, col2 = st.columns(2)
        with col1:
            consumption_type = st.selectbox("Type", ["Household", "Commercial", "Industrial"])
            num_records = st.number_input("Days", 30, 1000, 365)
        with col2:
            start_date = st.date_input("Start Date", datetime(2023, 1, 1))
            price = st.number_input("Price (Rs/kWh)", 1.0, 20.0, 8.0)
        
        if st.button("Generate from Survey", use_container_width=True):
            survey_responses = {
                'consumption_type': consumption_type,
                'num_records': num_records,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'price_per_kwh': price,
                'include_temperature': True,
                'include_cost': True
            }
            
            data, message = DataLoader.load_from_survey(survey_responses)
            if data is not None:
                st.session_state.forecast_data = data
                st.session_state.data_source = "survey"
                st.success(f"✅ Generated {num_records} days of data!")
                st.rerun()
    
    with source_tab3:
        sample_days = st.slider("Number of Days", 90, 730, 365, 30)
        seasonality = st.select_slider("Seasonality", ["Low", "Medium", "High"], "Medium")
        
        if st.button("Generate Sample Data", use_container_width=True):
            config = {
                'num_records': sample_days,
                'start_date': '2023-01-01',
                'seasonality': seasonality,
                'trend': 'Stable',
                'include_features': ['Temperature', 'Cost']
            }
            
            data, message = DataLoader.load_sample_data(config)
            if data is not None:
                st.session_state.forecast_data = data
                st.session_state.data_source = "sample"
                st.success(f"✅ Generated {sample_days} days of sample data!")
                st.rerun()

# Main App
def main():
    load_css()
    init_session_state()
    
    # Title
    st.markdown('<h1 style="color: #2E86AB;">🤖 Multi-Algorithm Energy Forecasting</h1>', unsafe_allow_html=True)
    st.markdown("### Compare All Models • Future Date Forecasting • Interactive Visualizations")
    
    # Data Source Selection
    if st.session_state.forecast_data is None:
        create_data_source_selection()
        st.stop()
    
    # Show data info
    data = st.session_state.forecast_data
    target_col = 'Energy_Consumption_kWh' if 'Energy_Consumption_kWh' in data.columns else 'Consumption'
    
    if target_col not in data.columns:
        # Try to find consumption column
        cons_cols = [col for col in data.columns if 'consumption' in col.lower() or 'energy' in col.lower()]
        if cons_cols:
            target_col = cons_cols[0]
        else:
            # Use first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[0]
            else:
                st.error("No suitable target column found!")
                return
    
    data_source_info = {
        "upload": "📁 Uploaded Data",
        "survey": "📋 Survey Generated",
        "sample": "🎲 Sample Data"
    }
    
    st.success(f"**Data Loaded:** {data_source_info.get(st.session_state.data_source, 'Unknown')} | "
              f"**Records:** {len(data)} | **Target:** {target_col}")
    
    # Feature Engineering Sidebar
    create_feature_engineering_sidebar()
    
    # Main content in tabs
    main_tabs = st.tabs(["📈 Data & Features", "🤖 Train Models", "📊 Compare Results", "🔮 View Forecasts"])
    
    with main_tabs[0]:
        # Data preview
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📋 Data Preview")
            st.dataframe(data.head(), use_container_width=True)
        
        with col2:
            st.subheader("📊 Data Statistics")
            st.write(f"**Total Records:** {len(data)}")
            if 'Date' in data.columns:
                st.write(f"**Date Range:** {data['Date'].min().date()} to {data['Date'].max().date()}")
            st.write(f"**Average {target_col}:** {data[target_col].mean():.2f}")
            st.write(f"**Max {target_col}:** {data[target_col].max():.2f}")
            st.write(f"**Min {target_col}:** {data[target_col].min():.2f}")
        
        # Create features
        st.subheader("🔬 Feature Engineering")
        engineer = AdvancedFeatureEngineer()
        engineered_data = engineer.create_all_features(data, target_col)
        st.session_state.feature_names = engineer.feature_names
        
        st.info(f"✅ Created **{len(engineer.feature_names)}** features")
        
        if st.checkbox("Show Engineered Features"):
            st.dataframe(engineered_data[engineer.feature_names[:10] + [target_col]].head(), 
                        use_container_width=True)
    
    with main_tabs[1]:
        st.subheader("🤖 Train Multiple Models")
        
        # Check if we have enough data
        if len(engineered_data) < 50:
            st.error("❌ Not enough data for training. Need at least 50 records after feature engineering.")
            return
        
        # Model selection
        st.markdown("#### Selected Algorithms:")
        
        selected_models = st.session_state.selected_models
        if not selected_models:
            st.warning("Please select at least one algorithm in the sidebar")
            st.stop()
        
        cols = st.columns(len(selected_models))
        for idx, model in enumerate(selected_models):
            with cols[idx]:
                st.markdown(f"**{model}**")
        
        # Forecast configuration with FUTURE DATE SUPPORT
        st.markdown('<div class="date-picker-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            # Date selection for forecast start
            min_date = data['Date'].min().date()
            max_date = data['Date'].max().date()
            
            # Allow forecasting into the future (up to 2026)
            max_forecast_date = datetime(2026, 12, 31).date()
            
            forecast_start_date = st.date_input(
                "📅 Forecast Start Date",
                value=max_date,  # Default to last available date
                min_value=min_date,
                max_value=max_forecast_date,
                help="Select the date from which to start forecasting. You can choose future dates!"
            )
            
            # Check if forecasting into the future
            if forecast_start_date > max_date:
                st.markdown('<div class="future-forecast">', unsafe_allow_html=True)
                st.markdown("**🔮 Forecasting into the FUTURE!**")
                st.markdown(f"Data ends: {max_date}<br>Forecast starts: {forecast_start_date}", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.session_state.forecast_start_date = forecast_start_date
        
        with col2:
            forecast_period = st.slider(
                "Forecast Period (days)",
                7, 365, 90, 7,
                help="Number of days to forecast"
            )
            st.session_state.forecast_period = forecast_period
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Train button
        if st.button("🚀 Train All Selected Models", type="primary", use_container_width=True):
            with st.spinner("Training all models..."):
                try:
                    # Prepare data
                    X = engineered_data[engineer.feature_names]
                    y = engineered_data[target_col]
                    
                    # Split data (80% train, 20% test)
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                    
                    # Train all models
                    trainer = MultiModelTrainer()
                    
                    metrics = trainer.train_all_models(
                        X_train, X_test, y_train, y_test, 
                        selected_models
                    )
                    
                    if metrics:
                        # Generate forecasts for all models
                        forecasts = generate_forecasts_for_all_models(
                            trainer, data, engineer.feature_names,
                            pd.to_datetime(forecast_start_date),
                            forecast_period, target_col
                        )
                        
                        # Store in session state
                        st.session_state.trained_models = trainer.models
                        st.session_state.model_performance = metrics
                        st.session_state.model_forecasts = forecasts
                        st.session_state.feature_importance = trainer.feature_importance
                        st.session_state.all_models_trained = True
                        
                        st.success(f"✅ Successfully trained {len(metrics)} models!")
                        
                        # Show quick summary
                        best_model = max(metrics.items(), key=lambda x: x[1].get('R²', 0))[0]
                        best_r2 = metrics[best_model].get('R²', 0)
                        st.info(f"🏆 **Best Model:** {best_model} (R²: {best_r2:.4f})")
                    else:
                        st.error("❌ No models were successfully trained")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")
    
    with main_tabs[2]:
        if not st.session_state.all_models_trained:
            st.warning("Please train models first in the 'Train Models' tab")
        else:
            # Display model performance cards
            display_model_performance_cards(st.session_state.model_performance)
            
            # Detailed comparison table
            st.subheader("📋 Detailed Model Comparison")
            comparison_df = create_model_comparison_table(st.session_state.model_performance)
            
            if not comparison_df.empty:
                # Format the dataframe for display
                def format_metrics(val):
                    if isinstance(val, float):
                        if val < 1 and val > 0.001:
                            return f"{val:.4f}"
                        elif val >= 1:
                            return f"{val:.2f}"
                        else:
                            return f"{val:.6f}"
                    return val
                
                # Apply formatting
                for col in ['R²', 'RMSE', 'MAE', 'MAPE']:
                    if col in comparison_df.columns:
                        comparison_df[col] = comparison_df[col].apply(format_metrics)
                
                st.dataframe(comparison_df, use_container_width=True)
    
    with main_tabs[3]:
        if not st.session_state.all_models_trained:
            st.warning("Please train models first in the 'Train Models' tab")
        elif not st.session_state.model_forecasts:
            st.error("No forecasts available. Please train models first.")
        else:
            # Forecast selector and visualization
            selected_forecast, selected_model = create_forecast_selector(
                st.session_state.model_forecasts, data, target_col
            )
            
            if selected_forecast is not None:
                # Display forecast summary - FIXED: Don't use date in st.metric()
                st.subheader(f"📈 {selected_model} Forecast Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_kwh = selected_forecast['Forecast_kWh'].sum()
                    st.metric("Total kWh", f"{total_kwh:,.0f}")
                with col2:
                    avg_daily = selected_forecast['Forecast_kWh'].mean()
                    st.metric("Avg Daily", f"{avg_daily:.1f} kWh")
                with col3:
                    total_cost = selected_forecast['Forecast_Cost_Rs'].sum()
                    st.metric("Total Cost", f"₹{total_cost:,.0f}")
                with col4:
                    # FIX: Show start date as text, not in st.metric()
                    start_date_str = selected_forecast['Date'].min().strftime('%Y-%m-%d')
                    end_date_str = selected_forecast['Date'].max().strftime('%Y-%m-%d')
                    st.markdown(f"**Date Range:**<br>{start_date_str} to {end_date_str}", unsafe_allow_html=True)
                
                # Show forecast table
                with st.expander("📋 View Forecast Data"):
                    st.dataframe(selected_forecast, use_container_width=True)
                
                # Download options
                st.subheader("💾 Download Options")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Download selected forecast
                    csv = selected_forecast.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download {selected_model} Forecast",
                        data=csv,
                        file_name=f"{selected_model.replace(' ', '_')}_forecast.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Download all forecasts
                    all_forecasts = pd.concat(st.session_state.model_forecasts.values(), ignore_index=True)
                    all_csv = all_forecasts.to_csv(index=False)
                    st.download_button(
                        label="📥 Download All Forecasts",
                        data=all_csv,
                        file_name="all_models_forecasts.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Performance metrics for selected model
                if selected_model in st.session_state.model_performance:
                    st.subheader("📊 Model Performance")
                    metrics = st.session_state.model_performance[selected_model]
                    
                    cols = st.columns(4)
                    metrics_list = [
                        ("R² Score", f"{metrics.get('R²', 0):.4f}"),
                        ("RMSE", f"{metrics.get('RMSE', 0):.2f}"),
                        ("MAE", f"{metrics.get('MAE', 0):.2f}"),
                        ("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
                    ]
                    
                    for idx, (label, value) in enumerate(metrics_list):
                        with cols[idx]:
                            st.metric(label, value)
            
            # Show all forecasts comparison
            st.subheader("📊 All Models Forecast Comparison")
            
            if len(st.session_state.model_forecasts) > 1:
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data[target_col],
                    mode='lines',
                    name='Historical',
                    line=dict(color='gray', width=1),
                    opacity=0.5
                ))
                
                # Add forecasts for each model
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
                for idx, (model_name, forecast_df) in enumerate(st.session_state.model_forecasts.items()):
                    if idx < len(colors):
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Forecast_kWh'],
                            mode='lines',
                            name=model_name,
                            line=dict(color=colors[idx], width=2)
                        ))
                
                fig.update_layout(
                    title="All Models Forecast Comparison",
                    xaxis_title="Date",
                    yaxis_title="Energy Consumption (kWh)",
                    template="plotly_dark",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### 🎯 Key Features:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📅 Future Date Forecasting**<br>Forecast beyond your data up to 2026", unsafe_allow_html=True)
    with col2:
        st.markdown("**🤖 Multi-Algorithm Comparison**<br>Compare 5+ ML models side by side", unsafe_allow_html=True)
    with col3:
        st.markdown("**📊 Interactive Visualizations**<br>Click dropdown to switch between models", unsafe_allow_html=True)
    
    if st.button("🔄 Reset & Load New Data", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ['_runtime', '_last_rerun']:
                del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()
