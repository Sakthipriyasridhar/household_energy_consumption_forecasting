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
        
        /* Sidebar text */
        .sidebar-text {
            color: #1a1a2e !important;
        }
        
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
        
        /* Download button styling */
        .download-btn {
            background: linear-gradient(45deg, #4CAF50, #2E7D32) !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
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
        "data_source": "sample",
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
        "confidence_level": 95
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Advanced Feature Engineering
class AdvancedFeatureEngineer:
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
    
    def create_all_features(self, df, target_col='Energy_Consumption_kWh', config=None):
        """Create comprehensive feature set - FIXED TO EXCLUDE NON-NUMERIC COLUMNS"""
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
        
        # FIX: Store only numeric feature names (excluding target, date, and any non-numeric columns)
        exclude_cols = ['Date', target_col, 'Cost_Rs', 'Price_Rs_per_kWh', 'Source', 'Location', 'temp_category']
        exclude_cols.extend([col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])])
        
        # Only include numeric columns
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

# ML Model Trainer with Enhanced Metrics
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
            model, metrics = self._train_linear_regression(X_train, X_test, y_train, y_test, params)
        elif model_name == "Gradient Boosting":
            model, metrics = self._train_gradient_boosting(X_train, X_test, y_train, y_test, params)
        elif model_name == "ARIMA" and STATSMODELS_AVAILABLE:
            model, metrics = self._train_arima(y_train, y_test, params)
        else:
            raise ValueError(f"Model {model_name} not supported")
        
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
        """Train Linear Regression with polynomial features"""
        # Check if we have enough features
        if X_train.shape[1] < 2:
            # If not enough features, use simple linear regression
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return model, self._calculate_comprehensive_metrics(y_test, y_pred)
        
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        poly_scaler = StandardScaler()
        X_train_poly_scaled = poly_scaler.fit_transform(X_train_poly)
        X_test_poly_scaled = poly_scaler.transform(X_test_poly)
        
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_poly_scaled, y_train)
        
        y_pred = model.predict(X_test_poly_scaled)
        
        return (model, self._calculate_comprehensive_metrics(y_test, y_pred), 
                poly, poly_scaler)
    
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
    
    def cross_validate(self, X, y, model_name, n_splits=5):
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if model_name == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_name == "XGBoost":
                model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            cv_scores.append({
                'RMSE': np.sqrt(mean_squared_error(y_val, y_pred)),
                'MAE': mean_absolute_error(y_val, y_pred),
                'R2': r2_score(y_val, y_pred)
            })
        
        self.cv_scores[model_name] = cv_scores
        return cv_scores

# Generate Forecast with Confidence Intervals
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
                    # Simple uncertainty for ARIMA
                    uncertainty = prediction * 0.15 * (1 + (i / periods) * 0.5)
                else:
                    prediction = model.predict(X_pred_scaled)[0]
                    # Uncertainty based on distance from training data
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
                                      value=st.session_state.feature_engineering['lag_features'],
                                      help="Include past values as features")
            
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
                                          value=st.session_state.feature_engineering['rolling_features'],
                                          help="Include moving averages and other statistics")
            
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
                                       value=st.session_state.feature_engineering['date_features'],
                                       help="Include day, month, year etc.")
            
            # Cyclical features
            cyclical_features = st.checkbox("Cyclical Encoding",
                                           value=st.session_state.feature_engineering['cyclical_features'],
                                           help="Convert cyclical features to sine/cosine")
            
            # Interaction features
            interaction_features = st.checkbox("Interaction Features",
                                              value=st.session_state.feature_engineering['interaction_features'],
                                              help="Create interaction terms (e.g., temp*consumption)")
        
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
            value=st.session_state.get('confidence_level', 95),
            help="Confidence level for prediction intervals"
        )
        st.session_state.confidence_level = confidence_level

# Display Model Performance Metrics
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
            if k not in ['MAE %', 'RMSE %']  # Already shown
        }
        detailed_df = pd.DataFrame(list(detailed_metrics.items()),
                                  columns=['Metric', 'Value'])
        st.dataframe(detailed_df, use_container_width=True)

# Create Interactive Visualizations
def create_visualizations(data, forecast_data, target_col, model_name):
    """Create comprehensive interactive visualizations"""
    # Create tabs for different visualizations
    viz_tabs = st.tabs(["📈 Forecast vs Actual", "📊 Confidence Intervals", 
                       "🔍 Feature Importance", "📅 Seasonal Patterns"])
    
    with viz_tabs[0]:
        # Forecast vs Actual
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data[target_col],
            mode='lines',
            name='Historical',
            line=dict(color='#FF6B6B', width=2),
            opacity=0.8
        ))
        
        # Forecast
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
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[1]:
        # Confidence intervals
        fig = go.Figure()
        
        # Confidence interval band
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_data['Date'], forecast_data['Date'][::-1]]),
            y=pd.concat([forecast_data['Upper_Bound_kWh'], forecast_data['Lower_Bound_kWh'][::-1]]),
            fill='toself',
            fillcolor='rgba(0, 180, 216, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f"{st.session_state.confidence_level}% Confidence Interval"
        ))
        
        # Forecast line
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
        # Feature importance (if available)
        if hasattr(st.session_state, 'feature_importance') and st.session_state.feature_importance is not None:
            feature_importance = st.session_state.feature_importance
            
            if isinstance(feature_importance, dict) and model_name in feature_importance:
                importance_scores = feature_importance[model_name]
                
                if isinstance(importance_scores, np.ndarray) and len(importance_scores) > 0:
                    # Get feature names
                    if hasattr(st.session_state, 'feature_names'):
                        feature_names = st.session_state.feature_names
                    else:
                        feature_names = [f'Feature_{i}' for i in range(len(importance_scores))]
                    
                    # Create dataframe
                    importance_df = pd.DataFrame({
                        'Feature': feature_names[:len(importance_scores)],
                        'Importance': importance_scores
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    # Create bar chart
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
        # Seasonal patterns
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

# Download Section
def create_download_section(forecast_data, data, model_name, metrics):
    """Create comprehensive download section"""
    st.subheader("💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download forecast as CSV
        csv = forecast_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast",
            data=csv,
            file_name=f"energy_forecast_{model_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_forecast"
        )
    
    with col2:
        # Download model metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Metrics",
            data=metrics_csv,
            file_name=f"model_metrics_{model_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_metrics"
        )
    
    with col3:
        # Download complete report
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
            use_container_width=True,
            key="download_report"
        )
    
    # Display forecast summary
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
    st.markdown("### Feature Engineering • Model Metrics • Confidence Intervals • Interactive Visualizations")
    
    # Feature Engineering Sidebar
    create_feature_engineering_sidebar()
    
    # Data loading section (simplified for this example)
    if st.session_state.forecast_data is None:
        st.warning("⚠️ Please load data from the Data Loader page first")
        if st.button("📥 Go to Data Loading"):
            st.switch_page("pages/2_Data_Loader.py")
        return
    
    data = st.session_state.forecast_data
    target_col = 'Energy_Consumption_kWh' if 'Energy_Consumption_kWh' in data.columns else 'Consumption'
    
    # Show data preview
    with st.expander("📋 View Loaded Data Preview", expanded=False):
        st.write(f"Data shape: {data.shape}")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Show column types
        st.write("### Column Data Types")
        st.write(data.dtypes)
    
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
        st.write("Feature names:", engineer.feature_names[:20])  # Show first 20
        st.dataframe(engineered_data[engineer.feature_names + [target_col]].head(), 
                    use_container_width=True)
    
    # Model training section - FIXED TO USE ONLY NUMERIC FEATURES
    st.subheader("🤖 Model Training & Evaluation")
    
    model_options = ["Random Forest", "XGBoost", "LightGBM", "Linear Regression", "Gradient Boosting"]
    if STATSMODELS_AVAILABLE:
        model_options.append("ARIMA")
    
    selected_model = st.selectbox("Select Model", model_options, 
                                 index=model_options.index(st.session_state.selected_model) 
                                 if st.session_state.selected_model in model_options else 0)
    
    st.session_state.selected_model = selected_model
    
    # Add forecast period selection
    forecast_period = st.slider("Forecast Period (days)", 30, 365, 90, 30,
                               help="Number of days to forecast into the future")
    st.session_state.forecast_period = forecast_period
    
    if st.button("🚀 Train Model & Generate Forecast", type="primary", use_container_width=True):
        with st.spinner(f"Training {selected_model} model..."):
            try:
                # Prepare data - CRITICAL FIX: Use only numeric features
                if not engineer.feature_names:
                    st.error("❌ No numeric features available for training!")
                    st.info("Try enabling more feature engineering options in the sidebar")
                    return
                
                # Verify all features are numeric
                non_numeric_features = [col for col in engineer.feature_names 
                                       if not pd.api.types.is_numeric_dtype(engineered_data[col])]
                
                if non_numeric_features:
                    st.warning(f"Removing non-numeric features: {non_numeric_features}")
                    engineer.feature_names = [col for col in engineer.feature_names 
                                            if col not in non_numeric_features]
                
                if not engineer.feature_names:
                    st.error("❌ No valid numeric features after cleaning!")
                    return
                
                # Use only numeric features
                X = engineered_data[engineer.feature_names]
                y = engineered_data[target_col]
                
                # Debug info
                st.write(f"✅ Using {X.shape[1]} numeric features for training")
                
                # Split data
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Check if we have enough data
                if len(X_train) < 10 or len(X_test) < 5:
                    st.error("❌ Not enough data for training. Need at least 10 training samples and 5 test samples.")
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
                
                # Generate forecast with confidence intervals
                forecast_result = generate_forecast_with_ci(
                    st.session_state.trained_model,
                    data,
                    forecast_period,
                    target_col,
                    st.session_state.confidence_level
                )
                
                if forecast_result is not None:
                    st.session_state.forecast_result = forecast_result
                    st.success(f"✅ {selected_model} trained successfully! Generated {forecast_period}-day forecast.")
                else:
                    st.error("❌ Forecast generation failed!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")
    
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
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by Advanced Machine Learning with Feature Engineering*")

if __name__ == "__main__":
    main()
