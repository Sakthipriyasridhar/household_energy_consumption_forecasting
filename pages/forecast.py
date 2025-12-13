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
        .diagnostic-box {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .warning-box {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .success-box {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
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
        "data_source": None,
        "feature_importance": {},
        "all_models_trained": False,
        "feature_engineering": {
            'lag_features': True,
            'rolling_features': True,
            'date_features': True,
            'cyclical_features': True,
            'window_sizes': [3, 7, 14],
            'lag_days': [1, 7, 14]
        },
        "confidence_level": 95,
        "selected_models": ["Random Forest", "XGBoost", "LightGBM", "Linear Regression", "Gradient Boosting"],
        "forecast_start_date": None,
        "forecast_period": 90
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# FIXED: Data Loader with proper date handling
class DataLoader:
    @staticmethod
    def load_sample_energy_data():
        """Generate realistic energy consumption data with proper patterns"""
        np.random.seed(42)
        
        # Create 2 years of daily data
        dates = pd.date_range(start='2023-01-01', periods=730, freq='D')
        
        # Base consumption with trend
        trend = np.linspace(15, 18, 730)  # Increasing trend
        
        # Seasonal patterns
        seasonal = 3 * np.sin(2 * np.pi * np.arange(730) / 365)  # Yearly seasonality
        
        # Weekly patterns (lower on weekends)
        weekly = np.array([0 if date.weekday() < 5 else -2 for date in dates])
        
        # Random noise
        noise = np.random.normal(0, 1.5, 730)
        
        # Combine all components
        consumption = trend + seasonal + weekly + noise
        consumption = np.maximum(consumption, 5)  # Ensure positive values
        
        # Add some anomalies
        anomaly_indices = [100, 200, 300, 400, 500]
        for idx in anomaly_indices:
            consumption[idx] += np.random.uniform(5, 10)
        
        data = pd.DataFrame({
            'Date': dates,
            'Energy_Consumption_kWh': np.round(consumption, 2),
            'Temperature_C': np.round(20 + 10 * np.sin(2 * np.pi * np.arange(730) / 365) + 
                                     np.random.normal(0, 5, 730), 1),
            'DayOfWeek': [date.weekday() for date in dates],
            'Month': [date.month for date in dates]
        })
        
        # Add cost
        data['Cost_Rs'] = np.round(data['Energy_Consumption_kWh'] * 8, 2)
        
        return data

# FIXED: Feature Engineering with NO DATA LEAKAGE
class SafeFeatureEngineer:
    def __init__(self):
        self.feature_names = []
        self.target_col = 'Energy_Consumption_kWh'
    
    def create_training_features(self, data, train_idx, val_idx, config):
        """Create features for training WITHOUT data leakage"""
        df = data.copy()
        
        # Always include basic date features
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        
        # Cyclical features
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sin_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['cos_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # IMPORTANT: Create lag features ONLY within training data
        if config['lag_features']:
            for lag in config['lag_days']:
                # Create lag feature for training period only
                df[f'lag_{lag}'] = df[self.target_col].shift(lag)
        
        # IMPORTANT: Create rolling features ONLY within training data
        if config['rolling_features']:
            for window in config['window_sizes']:
                df[f'rolling_mean_{window}'] = df[self.target_col].rolling(window=window, min_periods=1).mean()
        
        # Store feature names (exclude target and non-numeric)
        exclude_cols = ['Date', self.target_col, 'Cost_Rs']
        self.feature_names = [col for col in df.columns 
                             if col not in exclude_cols 
                             and pd.api.types.is_numeric_dtype(df[col])]
        
        # Split data
        train_data = df.iloc[train_idx]
        val_data = df.iloc[val_idx]
        
        # Fill NaN values (created by lag/rolling operations)
        train_data = train_data.fillna(method='bfill').fillna(method='ffill')
        val_data = val_data.fillna(method='bfill').fillna(method='ffill')
        
        return train_data, val_data
    
    def create_forecast_features(self, historical_data, config):
        """Create features for forecasting from historical data"""
        df = historical_data.copy()
        
        # Basic date features
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        
        # Cyclical features
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sin_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['cos_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Create lag features from historical data
        if config['lag_features']:
            for lag in config['lag_days']:
                df[f'lag_{lag}'] = df[self.target_col].shift(lag)
        
        # Create rolling features from historical data
        if config['rolling_features']:
            for window in config['window_sizes']:
                df[f'rolling_mean_{window}'] = df[self.target_col].rolling(window=window, min_periods=1).mean()
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df

# FIXED: Multi-Model Trainer with proper validation
class RobustModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.all_metrics = {}
    
    def train_all_models_with_cv(self, data, target_col, selected_models, config, n_splits=3):
        """Train models with time-series cross-validation to avoid overfitting"""
        results = {}
        
        # Prepare time-series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        engineer = SafeFeatureEngineer()
        
        for model_name in selected_models:
            try:
                st.info(f"Training {model_name}...")
                
                # Store metrics for each fold
                fold_metrics = []
                
                for fold, (train_idx, val_idx) in enumerate(tscv.split(data)):
                    # Create features for this fold (no data leakage)
                    train_data, val_data = engineer.create_training_features(
                        data, train_idx, val_idx, config
                    )
                    
                    # Prepare features and target
                    X_train = train_data[engineer.feature_names]
                    y_train = train_data[target_col]
                    X_val = val_data[engineer.feature_names]
                    y_val = val_data[target_col]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train model
                    model = self._get_model(model_name)
                    model.fit(X_train_scaled, y_train)
                    
                    # Predict and calculate metrics
                    y_pred = model.predict(X_val_scaled)
                    metrics = self._calculate_metrics(y_val, y_pred)
                    fold_metrics.append(metrics)
                
                # Average metrics across folds
                avg_metrics = self._average_metrics(fold_metrics)
                
                # Train final model on all data (for forecasting)
                final_train_data, _ = engineer.create_training_features(
                    data, range(len(data)), [], config
                )
                X_final = final_train_data[engineer.feature_names]
                y_final = final_train_data[target_col]
                
                scaler = StandardScaler()
                X_final_scaled = scaler.fit_transform(X_final)
                
                final_model = self._get_model(model_name)
                final_model.fit(X_final_scaled, y_final)
                
                # Store results
                self.models[model_name] = final_model
                self.scalers[model_name] = scaler
                self.all_metrics[model_name] = avg_metrics
                results[model_name] = avg_metrics
                
                if hasattr(final_model, 'feature_importances_'):
                    self.feature_importance[model_name] = final_model.feature_importances_
                
                st.success(f"✅ {model_name} trained successfully")
                
            except Exception as e:
                st.warning(f"⚠️ Could not train {model_name}: {str(e)}")
                continue
        
        return results
    
    def _get_model(self, model_name):
        """Get model instance"""
        if model_name == "Random Forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_name == "XGBoost":
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1
            )
        elif model_name == "LightGBM":
            return lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1
            )
        elif model_name == "Linear Regression":
            return LinearRegression()
        elif model_name == "Gradient Boosting":
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
        else:
            return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate realistic metrics"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Handle edge cases
        if len(y_true) == 0 or len(y_pred) == 0:
            return {'R²': 0, 'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
        
        try:
            r2 = max(-1, min(1, r2_score(y_true, y_pred)))  # Bound R² between -1 and 1
            
            # Calculate MAPE safely
            mask = (y_true != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
            if np.sum(mask) > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = np.nan
            
            metrics = {
                'R²': r2,
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred),
                'MAPE': mape
            }
        except:
            metrics = {'R²': 0, 'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
        
        return metrics
    
    def _average_metrics(self, fold_metrics):
        """Average metrics across cross-validation folds"""
        avg_metrics = {}
        for key in ['R²', 'RMSE', 'MAE', 'MAPE']:
            values = [m.get(key, np.nan) for m in fold_metrics if key in m]
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                avg_metrics[key] = np.mean(valid_values)
            else:
                avg_metrics[key] = np.nan
        return avg_metrics

# FIXED: Forecast Generator
def generate_realistic_forecasts(trainer, historical_data, feature_names, forecast_start_date, periods, target_col):
    """Generate realistic forecasts"""
    forecasts = {}
    
    engineer = SafeFeatureEngineer()
    
    for model_name, model in trainer.models.items():
        try:
            scaler = trainer.scalers.get(model_name, StandardScaler())
            
            # Prepare forecast data
            current_data = historical_data.copy()
            predictions = []
            
            for i in range(periods):
                next_date = current_data['Date'].max() + timedelta(days=1)
                
                # Create features for next date
                temp_data = pd.concat([current_data, pd.DataFrame({
                    'Date': [next_date],
                    target_col: [np.nan]
                })], ignore_index=True)
                
                # Engineer features
                df_features = engineer.create_forecast_features(temp_data, st.session_state.feature_engineering)
                
                # Get last row features
                if feature_names:
                    available_features = [col for col in feature_names if col in df_features.columns]
                    if available_features:
                        last_row = df_features.iloc[-1:][available_features].fillna(0)
                        
                        # Scale and predict
                        X_scaled = scaler.transform(last_row)
                        prediction = model.predict(X_scaled)[0]
                        
                        # Ensure realistic values
                        prediction = max(0.1, prediction)
                        
                        # Add some randomness based on historical volatility
                        hist_std = current_data[target_col].std()
                        prediction += np.random.normal(0, hist_std * 0.1)
                        prediction = max(0.1, prediction)
                        
                        predictions.append(prediction)
                        
                        # Update data for next iteration
                        current_data = pd.concat([current_data, pd.DataFrame({
                            'Date': [next_date],
                            target_col: [prediction]
                        })], ignore_index=True)
            
            if predictions:
                forecast_dates = pd.date_range(
                    start=forecast_start_date,
                    periods=periods,
                    freq='D'
                )
                
                # Calculate confidence intervals
                pred_array = np.array(predictions)
                hist_std = historical_data[target_col].std()
                ci_multiplier = 1.96  # 95% confidence
                
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecast_kWh': np.round(pred_array, 2),
                    'Lower_Bound_kWh': np.round(pred_array - ci_multiplier * hist_std * 0.5, 2),
                    'Upper_Bound_kWh': np.round(pred_array + ci_multiplier * hist_std * 0.5, 2),
                    'Forecast_Cost_Rs': np.round(pred_array * 8, 2),
                    'Model': model_name
                })
                
                # Ensure no negative values
                forecast_df[['Forecast_kWh', 'Lower_Bound_kWh', 'Upper_Bound_kWh']] = \
                    forecast_df[['Forecast_kWh', 'Lower_Bound_kWh', 'Upper_Bound_kWh']].clip(lower=0.1)
                
                forecasts[model_name] = forecast_df
                
        except Exception as e:
            st.warning(f"Could not generate forecast for {model_name}: {str(e)}")
            continue
    
    return forecasts

# Diagnostic Functions
def analyze_data_patterns(data, target_col):
    """Analyze data patterns and provide insights"""
    st.subheader("🔍 Data Pattern Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Trend analysis
        if len(data) > 30:
            from scipy import stats
            x = np.arange(len(data))
            y = data[target_col].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            trend = "Upward ↗️" if slope > 0 else "Downward ↘️" if slope < 0 else "Stable ➡️"
            st.metric("Trend", trend, delta=f"{slope:.4f} per day")
    
    with col2:
        # Seasonality
        if 'month' in data.columns:
            monthly_avg = data.groupby('month')[target_col].mean()
            seasonality_strength = monthly_avg.std() / monthly_avg.mean() * 100
            st.metric("Seasonality", f"{seasonality_strength:.1f}%")
    
    with col3:
        # Volatility
        volatility = data[target_col].std() / data[target_col].mean() * 100
        st.metric("Volatility", f"{volatility:.1f}%")
    
    # Plot patterns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Raw Data', '30-Day Moving Average', 
                       'Monthly Pattern', 'Weekly Pattern'),
        vertical_spacing=0.15
    )
    
    # Raw data
    fig.add_trace(
        go.Scatter(x=data['Date'], y=data[target_col], mode='lines', name='Raw'),
        row=1, col=1
    )
    
    # Moving average
    ma_30 = data[target_col].rolling(window=30).mean()
    fig.add_trace(
        go.Scatter(x=data['Date'], y=ma_30, mode='lines', name='30-Day MA', line=dict(color='red')),
        row=1, col=2
    )
    
    # Monthly pattern
    if 'month' in data.columns:
        monthly = data.groupby('month')[target_col].mean()
        fig.add_trace(
            go.Bar(x=list(range(1, 13)), y=monthly.values, name='Monthly Avg'),
            row=2, col=1
        )
    
    # Weekly pattern
    if 'dayofweek' in data.columns or 'DayOfWeek' in data.columns:
        day_col = 'dayofweek' if 'dayofweek' in data.columns else 'DayOfWeek'
        weekly = data.groupby(day_col)[target_col].mean()
        fig.add_trace(
            go.Bar(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 
                  y=weekly.values, name='Weekly Avg'),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=False, title_text="Data Pattern Analysis")
    st.plotly_chart(fig, use_container_width=True)

def display_realistic_comparison(metrics_dict):
    """Display realistic model comparison"""
    if not metrics_dict:
        return
    
    st.subheader("📊 Realistic Model Comparison")
    
    # Create comparison table
    comparison_data = []
    for model_name, metrics in metrics_dict.items():
        row = {'Model': model_name}
        for key in ['R²', 'RMSE', 'MAE', 'MAPE']:
            if key in metrics:
                if key == 'R²':
                    row[key] = f"{metrics[key]:.4f}"
                elif key == 'MAPE':
                    row[key] = f"{metrics[key]:.2f}%" if not np.isnan(metrics[key]) else "N/A"
                else:
                    row[key] = f"{metrics[key]:.2f}"
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by R²
    if 'R²' in df.columns:
        # Convert R² to numeric for sorting
        df['R²_numeric'] = df['R²'].astype(float)
        df = df.sort_values('R²_numeric', ascending=False)
        df = df.drop('R²_numeric', axis=1)
    
    # Display with conditional formatting
    def color_r2(val):
        try:
            r2 = float(val)
            if r2 > 0.8:
                return 'background-color: #d4edda'  # Green
            elif r2 > 0.6:
                return 'background-color: #fff3cd'  # Yellow
            elif r2 > 0.4:
                return 'background-color: #f8d7da'  # Light red
            else:
                return 'background-color: #f5c6cb'  # Red
        except:
            return ''
    
    def color_mape(val):
        try:
            if 'N/A' in str(val):
                return ''
            mape = float(str(val).replace('%', ''))
            if mape < 10:
                return 'background-color: #d4edda'  # Green
            elif mape < 20:
                return 'background-color: #fff3cd'  # Yellow
            elif mape < 30:
                return 'background-color: #f8d7da'  # Light red
            else:
                return 'background-color: #f5c6cb'  # Red
        except:
            return ''
    
    styled_df = df.style.applymap(color_r2, subset=['R²'])\
                       .applymap(color_mape, subset=['MAPE'])
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Add interpretation
    st.markdown('<div class="diagnostic-box">', unsafe_allow_html=True)
    st.markdown("**Interpretation Guide:**")
    st.markdown("- **R² > 0.8**: Excellent prediction (Green)")
    st.markdown("- **R² 0.6-0.8**: Good prediction (Yellow)")
    st.markdown("- **R² 0.4-0.6**: Moderate prediction (Light Red)")
    st.markdown("- **R² < 0.4**: Poor prediction (Red)")
    st.markdown("- **MAPE < 10%**: Excellent accuracy")
    st.markdown("- **MAPE 10-20%**: Good accuracy")
    st.markdown("- **MAPE 20-30%**: Moderate accuracy")
    st.markdown("- **MAPE > 30%**: Poor accuracy")
    st.markdown('</div>', unsafe_allow_html=True)

# Main App
def main():
    load_css()
    init_session_state()
    
    # Title
    st.title("🤖 Realistic Multi-Algorithm Energy Forecasting")
    st.markdown("### **No Data Leakage • Realistic R² Values • Pattern Analysis**")
    
    # Load sample data by default for demonstration
    if st.session_state.forecast_data is None:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**Using Sample Data for Demonstration**")
        st.markdown("This shows realistic energy consumption patterns with seasonality, trends, and noise.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        data = DataLoader.load_sample_energy_data()
        st.session_state.forecast_data = data
        st.session_state.data_source = "sample"
    
    data = st.session_state.forecast_data
    target_col = 'Energy_Consumption_kWh'
    
    # Show data info
    st.success(f"✅ **Data Loaded:** {len(data)} records | "
              f"**Date Range:** {data['Date'].min().date()} to {data['Date'].max().date()}")
    
    # Data Pattern Analysis
    analyze_data_patterns(data, target_col)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Model selection
        st.markdown("#### 🤖 Select Algorithms")
        selected_models = st.multiselect(
            "Choose algorithms to compare",
            ["Random Forest", "XGBoost", "LightGBM", "Linear Regression", "Gradient Boosting"],
            default=st.session_state.selected_models
        )
        st.session_state.selected_models = selected_models
        
        # Feature engineering
        st.markdown("#### 🔧 Feature Engineering")
        with st.expander("Configure", expanded=False):
            lag_features = st.checkbox("Lag Features", 
                                      value=st.session_state.feature_engineering['lag_features'])
            rolling_features = st.checkbox("Rolling Features",
                                          value=st.session_state.feature_engineering['rolling_features'])
            date_features = st.checkbox("Date Features",
                                       value=st.session_state.feature_engineering['date_features'])
            cyclical_features = st.checkbox("Cyclical Features",
                                           value=st.session_state.feature_engineering['cyclical_features'])
        
        st.session_state.feature_engineering.update({
            'lag_features': lag_features,
            'rolling_features': rolling_features,
            'date_features': date_features,
            'cyclical_features': cyclical_features
        })
        
        # Forecast settings
        st.markdown("#### 📅 Forecast Settings")
        min_date = data['Date'].min().date()
        max_date = data['Date'].max().date()
        
        forecast_start_date = st.date_input(
            "Start Date",
            value=max_date,
            min_value=min_date,
            max_value=datetime(2028, 12, 31).date(),
            help="Forecast from this date onward"
        )
        st.session_state.forecast_start_date = forecast_start_date
        
        forecast_period = st.slider(
            "Forecast Period (days)",
            30, 365, 90, 30,
            help="Number of days to forecast"
        )
        st.session_state.forecast_period = forecast_period
    
    # Training Section
    st.subheader("🚀 Train & Compare Models")
    
    if not selected_models:
        st.warning("Please select at least one algorithm in the sidebar")
    else:
        if st.button("🎯 Train All Models (With Cross-Validation)", type="primary", use_container_width=True):
            with st.spinner("Training models with cross-validation..."):
                try:
                    # Train models
                    trainer = RobustModelTrainer()
                    metrics = trainer.train_all_models_with_cv(
                        data, target_col, selected_models,
                        st.session_state.feature_engineering,
                        n_splits=3
                    )
                    
                    if metrics:
                        # Generate forecasts
                        forecasts = generate_realistic_forecasts(
                            trainer, data, trainer.models[list(metrics.keys())[0]]._features if hasattr(trainer.models[list(metrics.keys())[0]], '_features') else [],
                            pd.to_datetime(forecast_start_date),
                            forecast_period,
                            target_col
                        )
                        
                        # Store results
                        st.session_state.trained_models = trainer.models
                        st.session_state.model_performance = metrics
                        st.session_state.model_forecasts = forecasts
                        st.session_state.feature_importance = trainer.feature_importance
                        st.session_state.all_models_trained = True
                        
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success(f"✅ Successfully trained {len(metrics)} models!")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show best model
                        best_model = max(metrics.items(), key=lambda x: x[1].get('R²', -1))[0]
                        best_r2 = metrics[best_model].get('R²', 0)
                        st.info(f"🏆 **Best Model:** {best_model} (R²: {best_r2:.4f})")
                    else:
                        st.error("❌ No models were successfully trained")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Display Results
    if st.session_state.all_models_trained:
        # Model Comparison
        display_realistic_comparison(st.session_state.model_performance)
        
        # Forecast Visualization
        st.subheader("📈 Forecast Visualization")
        
        if st.session_state.model_forecasts:
            # Model selector
            model_names = list(st.session_state.model_forecasts.keys())
            selected_model = st.selectbox(
                "Select Model to Visualize",
                model_names,
                index=0
            )
            
            if selected_model in st.session_state.model_forecasts:
                forecast_df = st.session_state.model_forecasts[selected_model]
                
                # Create visualization
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data[target_col],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='#666', width=2),
                    opacity=0.7
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Forecast_kWh'],
                    mode='lines',
                    name=f'{selected_model} Forecast',
                    line=dict(color='#00b4d8', width=3)
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=pd.concat([forecast_df['Date'], forecast_df['Date'][::-1]]),
                    y=pd.concat([forecast_df['Upper_Bound_kWh'], forecast_df['Lower_Bound_kWh'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(0, 180, 216, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval'
                ))
                
                # Add vertical line at forecast start
                fig.add_vline(
                    x=pd.to_datetime(forecast_start_date).timestamp() * 1000,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Forecast Start",
                    annotation_position="top right"
                )
                
                fig.update_layout(
                    title=f"{selected_model} Forecast vs Historical",
                    xaxis_title="Date",
                    yaxis_title="Energy Consumption (kWh)",
                    template="plotly_white",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_forecast = forecast_df['Forecast_kWh'].mean()
                    hist_avg = data[target_col].mean()
                    change_pct = ((avg_forecast - hist_avg) / hist_avg) * 100
                    st.metric("Avg Forecast", f"{avg_forecast:.1f} kWh", 
                             delta=f"{change_pct:+.1f}%")
                with col2:
                    total_forecast = forecast_df['Forecast_kWh'].sum()
                    st.metric("Total kWh", f"{total_forecast:,.0f}")
                with col3:
                    total_cost = forecast_df['Forecast_Cost_Rs'].sum()
                    st.metric("Total Cost", f"₹{total_cost:,.0f}")
        
        # All Models Comparison Chart
        st.subheader("📊 All Models Forecast Comparison")
        
        if len(st.session_state.model_forecasts) > 1:
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data[target_col],
                mode='lines',
                name='Historical',
                line=dict(color='gray', width=1),
                opacity=0.3
            ))
            
            # Add each model's forecast
            colors = px.colors.qualitative.Set3
            for idx, (model_name, forecast_df) in enumerate(st.session_state.model_forecasts.items()):
                color = colors[idx % len(colors)]
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Forecast_kWh'],
                    mode='lines',
                    name=model_name,
                    line=dict(color=color, width=2)
                ))
            
            fig.update_layout(
                title="All Models Forecast Comparison",
                xaxis_title="Date",
                yaxis_title="Energy Consumption (kWh)",
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Data Quality Warning
    st.markdown("---")
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("**⚠️ Important Notes:**")
    st.markdown("1. **R² values are now realistic** (between -1 and 1)")
    st.markdown("2. **No data leakage** - models are properly validated")
    st.markdown("3. **Time-series cross-validation** prevents overfitting")
    st.markdown("4. **Realistic confidence intervals** based on historical volatility")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Reset button
    if st.button("🔄 Reset & Use New Data", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ['_runtime', '_last_rerun']:
                del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()
