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
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        
        /* Data source cards */
        .data-source-card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 5px solid;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .upload-card { border-left-color: #4CAF50; }
        .survey-card { border-left-color: #2196F3; }
        .sample-card { border-left-color: #FF9800; }
        
        /* Feature engineering cards */
        .feature-card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #00b4d8;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #2c3e50, #4a6491);
            color: white;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            margin: 0.5rem 0;
        }
        
        /* Model performance cards */
        .performance-card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            border: 1px solid #e0e0e0;
        }
        
        /* Confidence interval styling */
        .ci-low { color: #FF6B6B; }
        .ci-high { color: #4CAF50; }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
        }
        
        /* Survey form styling */
        .survey-question {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Initialize Session State
def init_session_state():
    defaults = {
        "forecast_data": None,
        "trained_model": None,
        "model_performance": {},
        "selected_model": "Random Forest",
        "forecast_result": None,
        "data_source": None,
        "file_uploaded": False,
        "feature_importance": None,
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
        "sample_config": {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Data Loading Functions
class DataLoader:
    """Handle different data sources"""
    
    @staticmethod
    def load_from_upload(uploaded_file):
        """Load data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                return None, "Unsupported file format"
            
            # Try to detect date column
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                df[date_columns[0]] = pd.to_datetime(df[date_columns[0]], errors='coerce')
            
            return df, "Success"
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def load_from_survey(survey_responses):
        """Generate data from survey responses"""
        try:
            # Extract survey parameters
            num_records = survey_responses.get('num_records', 365)
            start_date = survey_responses.get('start_date', '2023-01-01')
            consumption_type = survey_responses.get('consumption_type', 'Household')
            
            # Generate date range
            dates = pd.date_range(start=start_date, periods=num_records, freq='D')
            
            # Base consumption based on type
            if consumption_type == 'Household':
                base_consumption = 15
                seasonal_factor = 5
            elif consumption_type == 'Commercial':
                base_consumption = 100
                seasonal_factor = 25
            else:  # Industrial
                base_consumption = 500
                seasonal_factor = 100
            
            # Generate synthetic data
            np.random.seed(42)
            
            # Create seasonal pattern
            seasonal = seasonal_factor * np.sin(2 * np.pi * np.arange(num_records) / 365)
            
            # Add weekday/weekend pattern
            weekday_factor = np.array([0.8 if date.weekday() < 5 else 1.2 for date in dates])
            
            # Add random noise
            noise = np.random.normal(0, base_consumption * 0.2, num_records)
            
            # Generate consumption values
            consumption = base_consumption + seasonal + weekday_factor * base_consumption * 0.1 + noise
            consumption = np.maximum(consumption, 0.1)  # Ensure positive
            
            # Add temperature data if available in survey
            if survey_responses.get('include_temperature', True):
                temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(num_records) / 365) + np.random.normal(0, 5, num_records)
            else:
                temperature = None
            
            # Create DataFrame
            data = pd.DataFrame({
                'Date': dates,
                'Energy_Consumption_kWh': np.round(consumption, 2)
            })
            
            if temperature is not None:
                data['Temperature_C'] = np.round(temperature, 1)
            
            # Add cost if applicable
            if survey_responses.get('include_cost', True):
                price_per_kwh = survey_responses.get('price_per_kwh', 8)
                data['Cost_Rs'] = np.round(data['Energy_Consumption_kWh'] * price_per_kwh, 2)
                data['Price_Rs_per_kWh'] = price_per_kwh
            
            # Add appliance breakdown if requested
            if survey_responses.get('appliance_breakdown', False):
                data['AC_Consumption'] = np.round(data['Energy_Consumption_kWh'] * 0.4, 2)
                data['Lighting_Consumption'] = np.round(data['Energy_Consumption_kWh'] * 0.2, 2)
                data['Other_Consumption'] = np.round(data['Energy_Consumption_kWh'] * 0.4, 2)
            
            return data, "Success"
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def load_sample_data(config=None):
        """Generate sample data for demonstration"""
        if config is None:
            config = {
                'num_records': 365,
                'start_date': '2023-01-01',
                'seasonality': 'High',
                'trend': 'Upward',
                'noise_level': 'Medium',
                'include_features': ['Temperature', 'Cost', 'Appliance_Breakdown']
            }
        
        try:
            num_records = config['num_records']
            dates = pd.date_range(start=config['start_date'], periods=num_records, freq='D')
            
            # Set parameters based on configuration
            if config['seasonality'] == 'High':
                seasonal_factor = 8
            elif config['seasonality'] == 'Medium':
                seasonal_factor = 5
            else:  # Low
                seasonal_factor = 2
            
            if config['trend'] == 'Upward':
                trend_factor = 0.02
            elif config['trend'] == 'Downward':
                trend_factor = -0.01
            else:  # Stable
                trend_factor = 0
            
            if config['noise_level'] == 'High':
                noise_factor = 0.3
            elif config['noise_level'] == 'Medium':
                noise_factor = 0.2
            else:  # Low
                noise_factor = 0.1
            
            # Generate base consumption with trend
            base_consumption = 20 + trend_factor * np.arange(num_records)
            
            # Add seasonality
            seasonal = seasonal_factor * np.sin(2 * np.pi * np.arange(num_records) / 365)
            
            # Add weekly pattern
            weekly_pattern = 3 * np.array([0.8 if date.weekday() < 5 else 1.2 for date in dates])
            
            # Add noise
            noise = noise_factor * base_consumption * np.random.randn(num_records)
            
            # Combine all components
            consumption = base_consumption + seasonal + weekly_pattern + noise
            consumption = np.maximum(consumption, 0.1)
            
            # Create DataFrame
            data = pd.DataFrame({
                'Date': dates,
                'Energy_Consumption_kWh': np.round(consumption, 2)
            })
            
            # Add additional features based on configuration
            if 'Temperature' in config['include_features']:
                data['Temperature_C'] = np.round(20 + 10 * np.sin(2 * np.pi * np.arange(num_records) / 365) + 
                                                np.random.normal(0, 5, num_records), 1)
            
            if 'Cost' in config['include_features']:
                data['Cost_Rs'] = np.round(data['Energy_Consumption_kWh'] * 8, 2)
                data['Price_Rs_per_kWh'] = 8
            
            if 'Appliance_Breakdown' in config['include_features']:
                data['AC_Consumption'] = np.round(data['Energy_Consumption_kWh'] * np.random.uniform(0.3, 0.5, num_records), 2)
                data['Lighting_Consumption'] = np.round(data['Energy_Consumption_kWh'] * np.random.uniform(0.1, 0.3, num_records), 2)
                data['Other_Consumption'] = np.round(data['Energy_Consumption_kWh'] * 
                                                    (1 - data['AC_Consumption']/data['Energy_Consumption_kWh'] - 
                                                     data['Lighting_Consumption']/data['Energy_Consumption_kWh']), 2)
            
            # Add some outliers
            if config.get('add_outliers', False):
                outlier_indices = np.random.choice(num_records, size=int(num_records * 0.05), replace=False)
                data.loc[outlier_indices, 'Energy_Consumption_kWh'] *= np.random.uniform(1.5, 3, len(outlier_indices))
            
            return data, "Success"
        except Exception as e:
            return None, str(e)

# Advanced Feature Engineering (Keep existing class)
class AdvancedFeatureEngineer:
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
    
    def create_all_features(self, df, target_col='Energy_Consumption_kWh', config=None):
        """Create comprehensive feature set"""
        if config is None:
            config = st.session_state.feature_engineering
        
        df = df.copy()
        original_len = len(df)
        
        # Store original non-numeric columns to exclude later
        self.non_numeric_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        
        # 1. Basic Date Features
        if config['date_features']:
            df = self._add_date_features(df)
        
        # 2. Cyclical Features
        if config['cyclical_features']:
            df = self._add_cyclical_features(df)
        
        # 3. Lag Features
        if config['lag_features']:
            for lag in config['lag_days']:
                df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # 4. Rolling Statistics
        if config['rolling_features']:
            for window in config['window_sizes']:
                df = self._add_rolling_features(df, target_col, window)
        
        # 5. Difference Features
        df = self._add_difference_features(df, target_col)
        
        # 6. Interaction Features (if temperature available)
        if config['interaction_features'] and 'Temperature_C' in df.columns:
            df = self._add_interaction_features(df, target_col)
        
        # Drop NaN rows created by lag/rolling features
        df = df.dropna()
        
        # Store only numeric feature names
        exclude_cols = ['Date', target_col] + self.non_numeric_cols
        self.feature_names = [col for col in df.columns 
                             if col not in exclude_cols 
                             and pd.api.types.is_numeric_dtype(df[col])]
        
        st.info(f"✅ Created {len(self.feature_names)} features from {original_len} records")
        return df
    
    def _add_date_features(self, df):
        """Add date-based features"""
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['dayofyear'] = df['Date'].dt.dayofyear
        df['weekofyear'] = df['Date'].dt.isocalendar().week
        df['quarter'] = df['Date'].dt.quarter
        df['is_weekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)
        df['is_month_start'] = (df['Date'].dt.day == 1).astype(int)
        df['is_month_end'] = (df['Date'].dt.days_in_month == df['Date'].dt.day).astype(int)
        return df
    
    def _add_cyclical_features(self, df):
        """Add cyclical encoding for date features"""
        if 'dayofyear' in df.columns:
            df['sin_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
            df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
        
        if 'dayofweek' in df.columns:
            df['sin_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['cos_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        if 'month' in df.columns:
            df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
            df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_rolling_features(self, df, target_col, window):
        """Add rolling statistics"""
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
        df[f'rolling_min_{window}'] = df[target_col].rolling(window=window, min_periods=1).min()
        df[f'rolling_max_{window}'] = df[target_col].rolling(window=window, min_periods=1).max()
        df[f'rolling_median_{window}'] = df[target_col].rolling(window=window, min_periods=1).median()
        
        # Exponential moving averages
        df[f'ema_{window}'] = df[target_col].ewm(span=window, adjust=False).mean()
        
        return df
    
    def _add_difference_features(self, df, target_col):
        """Add difference features"""
        df['diff_1'] = df[target_col].diff(1)
        df['diff_7'] = df[target_col].diff(7)
        df['diff_30'] = df[target_col].diff(30)
        
        # Percentage changes
        df['pct_change_1'] = df[target_col].pct_change(1)
        df['pct_change_7'] = df[target_col].pct_change(7)
        
        return df
    
    def _add_interaction_features(self, df, target_col):
        """Add interaction features with temperature"""
        if 'Temperature_C' in df.columns and pd.api.types.is_numeric_dtype(df['Temperature_C']):
            df['temp_consumption_interaction'] = df[target_col] * df['Temperature_C']
            df['temp_squared'] = df['Temperature_C'] ** 2
        
        return df

# ML Model Trainer (Keep existing class - but FIX the linear regression method)
class MLModelTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.cv_scores = {}
    
    def train_model(self, X_train, X_test, y_train, y_test, model_name, params=None):
        """Train a specific ML model"""
        if model_name == "Random Forest":
            model, metrics = self._train_random_forest(X_train, X_test, y_train, y_test, params)
        elif model_name == "XGBoost":
            model, metrics = self._train_xgboost(X_train, X_test, y_train, y_test, params)
        elif model_name == "LightGBM":
            model, metrics = self._train_lightgbm(X_train, X_test, y_train, y_test, params)
        elif model_name == "Linear Regression":
            # FIXED: Only return model and metrics
            model_result = self._train_linear_regression(X_train, X_test, y_train, y_test, params)
            if isinstance(model_result, tuple) and len(model_result) >= 2:
                model, metrics = model_result[0], model_result[1]
            else:
                model, metrics = None, {"error": "Training failed"}
        elif model_name == "Gradient Boosting":
            model, metrics = self._train_gradient_boosting(X_train, X_test, y_train, y_test, params)
        elif model_name == "ARIMA" and STATSMODELS_AVAILABLE:
            model, metrics = self._train_arima(y_train, y_test, params)
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        if model is not None:
            self.models[model_name] = model
        return model, metrics
    
    def _train_random_forest(self, X_train, X_test, y_train, y_test, params):
        """Train Random Forest model"""
        if params is None:
            params = {'n_estimators': 200, 'max_depth': 15}
        
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', 15),
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Store feature importance
        self.feature_importance['Random Forest'] = model.feature_importances_
        
        return model, self._calculate_comprehensive_metrics(y_test, y_pred)
    
    def _train_xgboost(self, X_train, X_test, y_train, y_test, params):
        """Train XGBoost model"""
        if params is None:
            params = {'n_estimators': 200, 'learning_rate': 0.05}
        
        model = xgb.XGBRegressor(
            n_estimators=params.get('n_estimators', 200),
            max_depth=8,
            learning_rate=params.get('learning_rate', 0.05),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.feature_importance['XGBoost'] = model.feature_importances_
        
        return model, self._calculate_comprehensive_metrics(y_test, y_pred)
    
    def _train_lightgbm(self, X_train, X_test, y_train, y_test, params):
        """Train LightGBM model"""
        if params is None:
            params = {'n_estimators': 200, 'learning_rate': 0.05}
        
        model = lgb.LGBMRegressor(
            n_estimators=params.get('n_estimators', 200),
            max_depth=10,
            learning_rate=params.get('learning_rate', 0.05),
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.feature_importance['LightGBM'] = model.feature_importances_
        
        return model, self._calculate_comprehensive_metrics(y_test, y_pred)
    
    def _train_linear_regression(self, X_train, X_test, y_train, y_test, params):
        """FIXED: Train Linear Regression - only return 2 values"""
        try:
            if X_train.shape[1] < 2:
                # Simple linear regression
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                return model, self._calculate_comprehensive_metrics(y_test, y_pred)
            
            # Polynomial features
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)
            
            # Scale
            poly_scaler = StandardScaler()
            X_train_poly_scaled = poly_scaler.fit_transform(X_train_poly)
            X_test_poly_scaled = poly_scaler.transform(X_test_poly)
            
            # Train Ridge regression
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_train_poly_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_poly_scaled)
            
            # Return only model and metrics (2 values)
            return model, self._calculate_comprehensive_metrics(y_test, y_pred)
            
        except Exception as e:
            return None, {"error": str(e)}
    
    def _train_gradient_boosting(self, X_train, X_test, y_train, y_test, params):
        """Train Gradient Boosting model"""
        if params is None:
            params = {'n_estimators': 200, 'learning_rate': 0.05}
        
        model = GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 200),
            learning_rate=params.get('learning_rate', 0.05),
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.feature_importance['Gradient Boosting'] = model.feature_importances_
        
        return model, self._calculate_comprehensive_metrics(y_test, y_pred)
    
    def _train_arima(self, y_train, y_test, params):
        """Train ARIMA model"""
        if not STATSMODELS_AVAILABLE:
            return None, {"error": "Statsmodels not installed"}
        
        try:
            order = params.get('order', (5, 1, 2)) if params else (5, 1, 2)
            model = ARIMA(y_train, order=order)
            model_fit = model.fit()
            
            forecast_steps = len(y_test)
            y_pred = model_fit.forecast(steps=forecast_steps)
            
            if len(y_pred) > len(y_test):
                y_pred = y_pred[:len(y_test)]
            elif len(y_pred) < len(y_test):
                y_test = y_test[:len(y_pred)]
            
            return model_fit, self._calculate_comprehensive_metrics(y_test, y_pred)
        except Exception as e:
            return None, {"error": str(e)}
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred):
        """Calculate comprehensive performance metrics"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Ensure no zeros for MAPE
        mask = y_true != 0
        if np.any(mask):
            y_true_masked = y_true[mask]
            y_pred_masked = y_pred[mask]
            mape = mean_absolute_percentage_error(y_true_masked, y_pred_masked) * 100
        else:
            mape = np.nan
        
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': mape,
            'Max Error': np.max(np.abs(y_true - y_pred)),
            'Mean Bias': np.mean(y_pred - y_true),
            'Std Error': np.std(y_pred - y_true),
            'Explained Variance': 1 - (np.var(y_true - y_pred) / np.var(y_true)) if np.var(y_true) > 0 else np.nan
        }
        
        # Add percentage errors
        if np.mean(y_true) > 0:
            metrics['MAE %'] = (metrics['MAE'] / np.mean(y_true)) * 100
            metrics['RMSE %'] = (metrics['RMSE'] / np.mean(y_true)) * 100
        else:
            metrics['MAE %'] = np.nan
            metrics['RMSE %'] = np.nan
        
        return metrics

# Generate Forecast with Confidence Intervals (Keep existing function)
def generate_forecast_with_ci(model_info, data, periods, target_col, confidence_level=95):
    """Generate forecast with confidence intervals"""
    model_type = model_info.get('model_type', 'Random Forest')
    model = model_info.get('model')
    scaler = model_info.get('scaler')
    feature_cols = model_info.get('feature_cols', [])
    
    if model is None:
        return None
    
    forecasts = []
    lower_bounds = []
    upper_bounds = []
    
    # Calculate z-score for confidence level
    ci_z_scores = {90: 1.645, 95: 1.96, 99: 2.576}
    z_score = ci_z_scores.get(confidence_level, 1.96)
    
    current_data = data.copy()
    
    for i in range(periods):
        next_date = current_data['Date'].max() + timedelta(days=1)
        
        # Create feature vector
        temp_combined = pd.concat([current_data, pd.DataFrame({
            'Date': [next_date],
            target_col: [np.nan]
        })], ignore_index=True)
        
        # Engineer features
        engineer = AdvancedFeatureEngineer()
        df_features = engineer.create_all_features(temp_combined, target_col, 
                                                  st.session_state.feature_engineering)
        
        if len(df_features) > 0:
            # Get only the feature columns that exist
            available_features = [col for col in feature_cols if col in df_features.columns]
            
            if available_features:
                last_features = df_features.iloc[-1:][available_features].fillna(method='ffill').fillna(0)
                
                if scaler:
                    X_pred_scaled = scaler.transform(last_features)
                else:
                    X_pred_scaled = last_features
                
                # Make prediction
                if model_type == 'ARIMA':
                    prediction = model.forecast(steps=1)[0]
                    uncertainty = prediction * 0.15 * (1 + (i / periods) * 0.5)
                else:
                    prediction = model.predict(X_pred_scaled)[0]
                    base_uncertainty = prediction * 0.12
                    time_uncertainty = (i / periods) * prediction * 0.08
                    uncertainty = base_uncertainty + time_uncertainty
                
                forecasts.append(max(0.1, prediction))
                lower_bounds.append(max(0.1, prediction - z_score * uncertainty))
                upper_bounds.append(max(0.1, prediction + z_score * uncertainty))
                
                # Update data
                current_data = pd.concat([current_data, pd.DataFrame({
                    'Date': [next_date],
                    target_col: [prediction]
                })], ignore_index=True)
    
    # Create forecast dataframe
    if forecasts:
        forecast_dates = pd.date_range(
            start=data['Date'].max() + timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast_kWh': np.round(forecasts, 2),
            'Lower_Bound_kWh': np.round(lower_bounds, 2),
            'Upper_Bound_kWh': np.round(upper_bounds, 2),
            'Forecast_Cost_Rs': np.round(np.array(forecasts) * 8, 2),
            'Lower_Cost_Rs': np.round(np.array(lower_bounds) * 8, 2),
            'Upper_Cost_Rs': np.round(np.array(upper_bounds) * 8, 2),
            'Confidence_Level': f"{confidence_level}%"
        })
        
        return forecast_df
    return None

# Feature Engineering Sidebar
def create_feature_engineering_sidebar():
    """Sidebar for feature engineering configuration"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔧 Feature Engineering")
        
        with st.expander("Configure Features", expanded=False):
            # Lag features
            lag_features = st.checkbox("Lag Features", 
                                      value=st.session_state.feature_engineering['lag_features'])
            
            if lag_features:
                lag_days = st.multiselect(
                    "Lag Days",
                    options=[1, 2, 3, 7, 14, 30, 60, 90],
                    default=st.session_state.feature_engineering['lag_days']
                )
            else:
                lag_days = []
            
            # Rolling features
            rolling_features = st.checkbox("Rolling Statistics",
                                          value=st.session_state.feature_engineering['rolling_features'])
            
            if rolling_features:
                window_sizes = st.multiselect(
                    "Window Sizes",
                    options=[3, 7, 14, 30, 60, 90],
                    default=st.session_state.feature_engineering['window_sizes']
                )
            else:
                window_sizes = []
            
            # Date features
            date_features = st.checkbox("Date Features",
                                       value=st.session_state.feature_engineering['date_features'])
            
            # Cyclical features
            cyclical_features = st.checkbox("Cyclical Encoding",
                                           value=st.session_state.feature_engineering['cyclical_features'])
            
            # Interaction features
            interaction_features = st.checkbox("Interaction Features",
                                              value=st.session_state.feature_engineering['interaction_features'])
        
        # Update session state
        st.session_state.feature_engineering.update({
            'lag_features': lag_features,
            'rolling_features': rolling_features,
            'date_features': date_features,
            'cyclical_features': cyclical_features,
            'interaction_features': interaction_features,
            'window_sizes': window_sizes,
            'lag_days': lag_days
        })
        
        # Confidence level
        st.markdown("---")
        st.markdown("### 📊 Confidence Interval")
        confidence_level = st.select_slider(
            "Confidence Level",
            options=[90, 95, 99],
            value=st.session_state.get('confidence_level', 95)
        )
        st.session_state.confidence_level = confidence_level

# Data Source Selection
def create_data_source_selection():
    """Create data source selection interface"""
    st.markdown('<div class="data-source-card">', unsafe_allow_html=True)
    st.subheader("📥 Select Data Source")
    
    # Create tabs for different data sources
    source_tab1, source_tab2, source_tab3 = st.tabs(["📁 Upload Data", "📋 Survey Input", "🎲 Sample Data"])
    
    with source_tab1:
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.markdown("### Upload Your Data")
        
        uploaded_file = st.file_uploader(
            "Choose CSV, Excel, or JSON file",
            type=['csv', 'xlsx', 'xls', 'json'],
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Load Uploaded Data", use_container_width=True):
                with st.spinner("Loading data..."):
                    data, message = DataLoader.load_from_upload(uploaded_file)
                    if data is not None:
                        # Ensure Date column
                        date_cols = [col for col in data.columns if 'date' in col.lower()]
                        if date_cols:
                            data = data.rename(columns={date_cols[0]: 'Date'})
                        elif 'Date' not in data.columns:
                            data['Date'] = pd.date_range(start='2023-01-01', periods=len(data), freq='D')
                        
                        # Ensure consumption column
                        cons_cols = [col for col in data.columns if 'consumption' in col.lower() or 'energy' in col.lower()]
                        if cons_cols:
                            data = data.rename(columns={cons_cols[0]: 'Energy_Consumption_kWh'})
                        elif 'Energy_Consumption_kWh' not in data.columns and len(data.columns) > 1:
                            # Use first numeric column as consumption
                            numeric_cols = data.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                data = data.rename(columns={numeric_cols[0]: 'Energy_Consumption_kWh'})
                        
                        st.session_state.forecast_data = data
                        st.session_state.data_source = "upload"
                        st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
                        st.rerun()
                    else:
                        st.error(f"❌ Error loading file: {message}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with source_tab2:
        st.markdown('<div class="survey-card">', unsafe_allow_html=True)
        st.markdown("### Energy Consumption Survey")
        
        with st.form("survey_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                consumption_type = st.selectbox(
                    "Consumption Type",
                    ["Household", "Commercial", "Industrial"],
                    help="Select the type of energy consumption"
                )
                
                num_records = st.number_input(
                    "Number of Days",
                    min_value=30,
                    max_value=1000,
                    value=365,
                    help="How many days of data to generate"
                )
            
            with col2:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime(2023, 1, 1),
                    help="Starting date for the data"
                )
                
                price_per_kwh = st.number_input(
                    "Electricity Price (Rs/kWh)",
                    min_value=1.0,
                    max_value=20.0,
                    value=8.0,
                    help="Cost per kWh"
                )
            
            st.markdown("### Additional Features")
            col3, col4 = st.columns(2)
            
            with col3:
                include_temperature = st.checkbox("Include Temperature Data", value=True)
                include_cost = st.checkbox("Include Cost Calculation", value=True)
            
            with col4:
                appliance_breakdown = st.checkbox("Include Appliance Breakdown", value=False)
                add_holidays = st.checkbox("Add Holiday Effects", value=True)
            
            submitted = st.form_submit_button("📊 Generate Data from Survey", use_container_width=True)
            
            if submitted:
                survey_responses = {
                    'consumption_type': consumption_type,
                    'num_records': num_records,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'price_per_kwh': price_per_kwh,
                    'include_temperature': include_temperature,
                    'include_cost': include_cost,
                    'appliance_breakdown': appliance_breakdown,
                    'add_holidays': add_holidays
                }
                
                with st.spinner("Generating data from survey..."):
                    data, message = DataLoader.load_from_survey(survey_responses)
                    if data is not None:
                        st.session_state.forecast_data = data
                        st.session_state.data_source = "survey"
                        st.session_state.survey_data = survey_responses
                        st.success(f"✅ Generated {num_records} days of {consumption_type} data!")
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {message}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with source_tab3:
        st.markdown('<div class="sample-card">', unsafe_allow_html=True)
        st.markdown("### Generate Sample Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_days = st.slider("Number of Days", 90, 730, 365, 30)
            seasonality = st.select_slider("Seasonality", ["Low", "Medium", "High"], value="Medium")
        
        with col2:
            trend = st.select_slider("Trend", ["Stable", "Upward", "Downward"], value="Stable")
            noise_level = st.select_slider("Noise Level", ["Low", "Medium", "High"], value="Medium")
        
        features = st.multiselect(
            "Include Additional Features",
            ["Temperature", "Cost", "Appliance_Breakdown"],
            default=["Temperature", "Cost"]
        )
        
        add_outliers = st.checkbox("Add Some Outliers", value=False)
        
        if st.button("🎲 Generate Sample Data", use_container_width=True):
            config = {
                'num_records': sample_days,
                'start_date': '2023-01-01',
                'seasonality': seasonality,
                'trend': trend,
                'noise_level': noise_level,
                'include_features': features,
                'add_outliers': add_outliers
            }
            
            with st.spinner("Generating sample data..."):
                data, message = DataLoader.load_sample_data(config)
                if data is not None:
                    st.session_state.forecast_data = data
                    st.session_state.data_source = "sample"
                    st.session_state.sample_config = config
                    st.success(f"✅ Generated {sample_days} days of sample data!")
                    st.rerun()
                else:
                    st.error(f"❌ Error: {message}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Display Model Performance Metrics (Keep existing function)
def display_model_performance(metrics, model_name):
    """Display comprehensive model performance metrics"""
    st.subheader(f"📈 {model_name} Performance Metrics")
    
    # Create tabs for different metric categories
    tab1, tab2, tab3 = st.tabs(["📊 Core Metrics", "📈 Error Analysis", "📋 Detailed Stats"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            r2 = metrics.get('R2', 0)
            r2_display = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
            delta_color = "normal" if r2 > 0.7 else "off"
            st.metric("R² Score", r2_display,
                     delta="Higher is better", 
                     delta_color=delta_color)
        
        with col2:
            mape = metrics.get('MAPE', np.nan)
            mape_display = f"{mape:.2f}%" if not np.isnan(mape) else "N/A"
            delta_status = "Excellent" if mape < 10 else "Good" if mape < 20 else "Fair" if mape < 30 else "Poor"
            st.metric("MAPE", mape_display,
                     delta=delta_status,
                     delta_color="inverse")
        
        with col3:
            rmse = metrics.get('RMSE', 0)
            st.metric("RMSE", f"{rmse:.2f} kWh",
                     delta="Lower is better",
                     delta_color="inverse")
        
        with col4:
            mae = metrics.get('MAE', 0)
            st.metric("MAE", f"{mae:.2f} kWh",
                     delta="Lower is better",
                     delta_color="inverse")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Error distribution
            st.markdown("##### Error Distribution")
            error_stats = {
                'Mean Bias': metrics.get('Mean Bias', 0),
                'Std Error': metrics.get('Std Error', 0),
                'Max Error': metrics.get('Max Error', 0)
            }
            error_df = pd.DataFrame(list(error_stats.items()), 
                                   columns=['Metric', 'Value'])
            st.dataframe(error_df, use_container_width=True)
        
        with col2:
            # Percentage errors
            st.markdown("##### Percentage Errors")
            pct_errors = {
                'MAE %': metrics.get('MAE %', np.nan),
                'RMSE %': metrics.get('RMSE %', np.nan)
            }
            pct_df = pd.DataFrame(list(pct_errors.items()),
                                 columns=['Metric', 'Value'])
            st.dataframe(pct_df, use_container_width=True)
    
    with tab3:
        # Detailed metrics table
        st.markdown("##### Complete Metrics")
        detailed_metrics = {
            k: v for k, v in metrics.items() 
            if k not in ['MAE %', 'RMSE %']
        }
        detailed_df = pd.DataFrame(list(detailed_metrics.items()),
                                  columns=['Metric', 'Value'])
        st.dataframe(detailed_df, use_container_width=True)

# Create Interactive Visualizations (Keep existing function)
def create_visualizations(data, forecast_data, target_col, model_name):
    """Create comprehensive interactive visualizations"""
    viz_tabs = st.tabs(["📈 Forecast vs Actual", "📊 Confidence Intervals", 
                       "🔍 Feature Importance", "📅 Seasonal Patterns"])
    
    with viz_tabs[0]:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data[target_col],
            mode='lines',
            name='Historical',
            line=dict(color='#FF6B6B', width=2),
            opacity=0.8
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_data['Date'],
            y=forecast_data['Forecast_kWh'],
            mode='lines',
            name='Forecast',
            line=dict(color='#00b4d8', width=3)
        ))
        
        fig.update_layout(
            title=f"{model_name} Forecast vs Historical Data",
            xaxis_title="Date",
            yaxis_title="Energy Consumption (kWh)",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[1]:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_data['Date'], forecast_data['Date'][::-1]]),
            y=pd.concat([forecast_data['Upper_Bound_kWh'], forecast_data['Lower_Bound_kWh'][::-1]]),
            fill='toself',
            fillcolor='rgba(0, 180, 216, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f"{st.session_state.confidence_level}% Confidence Interval"
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_data['Date'],
            y=forecast_data['Forecast_kWh'],
            mode='lines',
            name='Forecast',
            line=dict(color='#00b4d8', width=3)
        ))
        
        fig.update_layout(
            title=f"Forecast with {st.session_state.confidence_level}% Confidence Intervals",
            xaxis_title="Date",
            yaxis_title="Energy Consumption (kWh)",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[2]:
        if hasattr(st.session_state, 'feature_importance') and st.session_state.feature_importance is not None:
            feature_importance = st.session_state.feature_importance
            
            if isinstance(feature_importance, dict) and model_name in feature_importance:
                importance_scores = feature_importance[model_name]
                
                if isinstance(importance_scores, np.ndarray) and len(importance_scores) > 0:
                    if hasattr(st.session_state, 'feature_names'):
                        feature_names = st.session_state.feature_names
                    else:
                        feature_names = [f'Feature_{i}' for i in range(len(importance_scores))]
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_names[:len(importance_scores)],
                        'Importance': importance_scores
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    fig = go.Figure(go.Bar(
                        x=importance_df['Importance'],
                        y=importance_df['Feature'],
                        orientation='h',
                        marker=dict(color='#4CAF50')
                    ))
                    
                    fig.update_layout(
                        title="Top 15 Feature Importance Scores",
                        xaxis_title="Importance Score",
                        yaxis_title="Feature",
                        template="plotly_dark",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance scores not available for this model")
            else:
                st.info("Feature importance analysis requires tree-based models")
        else:
            st.info("Train a model to see feature importance")
    
    with viz_tabs[3]:
        if 'month' in data.columns:
            monthly_avg = data.groupby('month')[target_col].mean().reset_index()
            
            fig = go.Figure(go.Bar(
                x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                y=monthly_avg[target_col],
                marker_color='#FF9800'
            ))
            
            fig.update_layout(
                title="Average Monthly Consumption Pattern",
                xaxis_title="Month",
                yaxis_title="Average Consumption (kWh)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add date features to see seasonal patterns")

# Download Section (Keep existing function)
def create_download_section(forecast_data, data, model_name, metrics):
    """Create comprehensive download section"""
    st.subheader("💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = forecast_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast",
            data=csv,
            file_name=f"energy_forecast_{model_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        metrics_df = pd.DataFrame([metrics])
        metrics_csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Metrics",
            data=metrics_csv,
            file_name=f"model_metrics_{model_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        report = f"""
        Energy Forecasting Report
        =========================
        
        Model: {model_name}
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Performance Metrics:
        -------------------
        R² Score: {metrics.get('R2', 0):.4f}
        MAPE: {metrics.get('MAPE', np.nan):.2f}%
        RMSE: {metrics.get('RMSE', 0):.2f} kWh
        MAE: {metrics.get('MAE', 0):.2f} kWh
        
        Forecast Summary:
        -----------------
        Total Forecast Period: {len(forecast_data)} days
        Average Daily Forecast: {forecast_data['Forecast_kWh'].mean():.2f} kWh
        Total Forecasted Consumption: {forecast_data['Forecast_kWh'].sum():.2f} kWh
        Total Estimated Cost: ₹{forecast_data['Forecast_Cost_Rs'].sum():.2f}
        
        Confidence Level: {st.session_state.confidence_level}%
        """
        
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name=f"forecasting_report_{model_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.markdown("---")
    st.markdown("#### 📋 Forecast Summary")
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric("Total kWh", f"{forecast_data['Forecast_kWh'].sum():,.0f}")
    
    with summary_col2:
        st.metric("Avg Daily", f"{forecast_data['Forecast_kWh'].mean():.1f} kWh")
    
    with summary_col3:
        st.metric("Total Cost", f"₹{forecast_data['Forecast_Cost_Rs'].sum():,.0f}")
    
    with summary_col4:
        ci_width = (forecast_data['Upper_Bound_kWh'] - forecast_data['Lower_Bound_kWh']).mean()
        st.metric("Avg CI Width", f"{ci_width:.1f} kWh")

# Main App
def main():
    load_css()
    init_session_state()
    
    # Title
    st.markdown('<h1 style="color: #2E86AB;">🤖 Advanced ML Energy Forecasting</h1>', unsafe_allow_html=True)
    st.markdown("### Multiple Data Sources • Feature Engineering • Model Comparison")
    
    # Data Source Selection (NEW - At the top)
    if st.session_state.forecast_data is None:
        create_data_source_selection()
        st.stop()
    
    # Show data source info
    data_source_info = {
        "upload": "📁 Uploaded Data",
        "survey": "📋 Survey Generated Data",
        "sample": "🎲 Sample Generated Data"
    }
    
    st.info(f"**Data Source:** {data_source_info.get(st.session_state.data_source, 'Unknown')}")
    
    # Feature Engineering Sidebar
    create_feature_engineering_sidebar()
    
    data = st.session_state.forecast_data
    target_col = 'Energy_Consumption_kWh' if 'Energy_Consumption_kWh' in data.columns else 'Consumption'
    
    # Show data preview
    with st.expander("📋 View Loaded Data", expanded=False):
        st.write(f"**Data Shape:** {data.shape}")
        st.write(f"**Date Range:** {data['Date'].min().date()} to {data['Date'].max().date()}")
        st.write(f"**Average Consumption:** {data[target_col].mean():.2f} kWh")
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(data.head(), use_container_width=True)
        with col2:
            st.dataframe(data.describe(), use_container_width=True)
    
    # Main content
    st.subheader("🔬 Advanced Feature Engineering")
    
    # Create features
    with st.spinner("Engineering features..."):
        engineer = AdvancedFeatureEngineer()
        engineered_data = engineer.create_all_features(data, target_col)
        st.session_state.feature_names = engineer.feature_names
    
    # Display feature statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Original Features:** {len(data.columns)}")
    with col2:
        st.info(f"**Engineered Features:** {len(engineer.feature_names)}")
    with col3:
        st.info(f"**Total Features:** {len(engineered_data.columns)}")
    
    # Show engineered features preview
    with st.expander("🔍 View Engineered Features", expanded=False):
        st.write(f"Available numeric features: {len(engineer.feature_names)}")
        st.write("First 20 feature names:", engineer.feature_names[:20])
        st.dataframe(engineered_data[engineer.feature_names + [target_col]].head(), 
                    use_container_width=True)
    
    # Model training section
    st.subheader("🤖 Model Training & Evaluation")
    
    model_options = ["Random Forest", "XGBoost", "LightGBM", "Linear Regression", "Gradient Boosting"]
    if STATSMODELS_AVAILABLE:
        model_options.append("ARIMA")
    
    selected_model = st.selectbox("Select Model", model_options, 
                                 index=model_options.index(st.session_state.selected_model) 
                                 if st.session_state.selected_model in model_options else 0)
    
    st.session_state.selected_model = selected_model
    
    # Add forecast period selection
    forecast_period = st.slider("Forecast Period (days)", 30, 365, 90, 30)
    st.session_state.forecast_period = forecast_period
    
    if st.button("🚀 Train Model & Generate Forecast", type="primary", use_container_width=True):
        with st.spinner(f"Training {selected_model} model..."):
            try:
                # Prepare data
                if not engineer.feature_names:
                    st.error("❌ No numeric features available for training!")
                    return
                
                # Use only numeric features
                X = engineered_data[engineer.feature_names]
                y = engineered_data[target_col]
                
                # Split data
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Check if we have enough data
                if len(X_train) < 10 or len(X_test) < 5:
                    st.error("❌ Not enough data for training.")
                    return
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                trainer = MLModelTrainer()
                model, metrics = trainer.train_model(
                    X_train_scaled, X_test_scaled, y_train, y_test,
                    selected_model, st.session_state.get('model_params')
                )
                
                if model is None:
                    st.error("❌ Model training failed!")
                    return
                
                # Store results
                st.session_state.trained_model = {
                    'model': model,
                    'scaler': scaler,
                    'feature_cols': engineer.feature_names,
                    'model_type': selected_model
                }
                st.session_state.model_performance = metrics
                st.session_state.feature_importance = trainer.feature_importance
                
                # Generate forecast
                forecast_result = generate_forecast_with_ci(
                    st.session_state.trained_model,
                    data,
                    forecast_period,
                    target_col,
                    st.session_state.confidence_level
                )
                
                if forecast_result is not None:
                    st.session_state.forecast_result = forecast_result
                    st.success(f"✅ {selected_model} trained successfully!")
                else:
                    st.error("❌ Forecast generation failed!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Display results if available
    if (st.session_state.trained_model is not None and 
        st.session_state.forecast_result is not None):
        
        # Display performance metrics
        display_model_performance(st.session_state.model_performance, 
                                 st.session_state.selected_model)
        
        # Create visualizations
        create_visualizations(data, st.session_state.forecast_result, 
                             target_col, st.session_state.selected_model)
        
        # Download section
        create_download_section(st.session_state.forecast_result, data,
                               st.session_state.selected_model,
                               st.session_state.model_performance)
        
        # Data source info
        st.markdown("---")
        st.markdown("#### 📊 Data Source Information")
        
        if st.session_state.data_source == "upload":
            st.info("Data loaded from uploaded file")
        elif st.session_state.data_source == "survey":
            st.info("Data generated from survey responses")
            with st.expander("View Survey Configuration"):
                st.json(st.session_state.survey_data)
        elif st.session_state.data_source == "sample":
            st.info("Sample data generated for demonstration")
            with st.expander("View Sample Configuration"):
                st.json(st.session_state.sample_config)
    
    # Footer with reset option
    st.markdown("---")
    if st.button("🔄 Load Different Data", use_container_width=True):
        st.session_state.forecast_data = None
        st.session_state.trained_model = None
        st.session_state.forecast_result = None
        st.rerun()
    
    st.markdown("*Powered by Advanced Machine Learning with Multiple Data Sources*")

if __name__ == "__main__":
    main()
