# pages/forecast.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# Time Series Models
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Note: Prophet requires separate install: pip install prophet

st.set_page_config(
    page_title="Energy Consumption Forecast",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .forecast-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .model-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid #e0e0e0;
        transition: transform 0.3s;
    }
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .metric-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8em;
        margin: 2px;
    }
    .badge-green { background: #d4edda; color: #155724; }
    .badge-yellow { background: #fff3cd; color: #856404; }
    .badge-red { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# ========== UTILITY FUNCTIONS ==========
def engineer_time_features(df, date_col='Date', target_col='Energy_Consumption_kWh'):
    """Engineer comprehensive time-based features to capture seasonality"""
    
    df_engineered = df.copy()
    
    # Ensure date is datetime
    df_engineered[date_col] = pd.to_datetime(df_engineered[date_col])
    
    # 1. TIME INDEX (Trend feature)
    df_engineered['time_index'] = np.arange(len(df_engineered))
    
    # 2. DATE COMPONENTS
    df_engineered['year'] = df_engineered[date_col].dt.year
    df_engineered['month'] = df_engineered[date_col].dt.month
    df_engineered['day'] = df_engineered[date_col].dt.day
    df_engineered['day_of_week'] = df_engineered[date_col].dt.dayofweek
    df_engineered['day_of_year'] = df_engineered[date_col].dt.dayofyear
    df_engineered['week_of_year'] = df_engineered[date_col].dt.isocalendar().week
    df_engineered['quarter'] = df_engineered[date_col].dt.quarter
    df_engineered['is_weekend'] = (df_engineered['day_of_week'] >= 5).astype(int)
    
    # 3. CYCLICAL FEATURES (SIN/COS) - MOST IMPORTANT FOR SEASONALITY!
    # Monthly seasonality
    df_engineered['month_sin'] = np.sin(2 * np.pi * df_engineered['month'] / 12)
    df_engineered['month_cos'] = np.cos(2 * np.pi * df_engineered['month'] / 12)
    
    # Weekly seasonality
    df_engineered['week_sin'] = np.sin(2 * np.pi * df_engineered['week_of_year'] / 52)
    df_engineered['week_cos'] = np.cos(2 * np.pi * df_engineered['week_of_year'] / 52)
    
    # Daily seasonality (within year)
    df_engineered['day_sin'] = np.sin(2 * np.pi * df_engineered['day_of_year'] / 365.25)
    df_engineered['day_cos'] = np.cos(2 * np.pi * df_engineered['day_of_year'] / 365.25)
    
    # Day of week seasonality
    df_engineered['dow_sin'] = np.sin(2 * np.pi * df_engineered['day_of_week'] / 7)
    df_engineered['dow_cos'] = np.cos(2 * np.pi * df_engineered['day_of_week'] / 7)
    
    # 4. LAG FEATURES (Historical patterns)
    for lag in [1, 2, 3, 7, 14, 30]:  # 1-day to 30-day lags
        df_engineered[f'lag_{lag}'] = df_engineered[target_col].shift(lag)
    
    # 5. ROLLING STATISTICS
    for window in [7, 14, 30, 90]:  # Weekly to quarterly patterns
        df_engineered[f'rolling_mean_{window}'] = df_engineered[target_col].rolling(window=window, min_periods=1).mean()
        df_engineered[f'rolling_std_{window}'] = df_engineered[target_col].rolling(window=window, min_periods=1).std()
        df_engineered[f'rolling_min_{window}'] = df_engineered[target_col].rolling(window=window, min_periods=1).min()
        df_engineered[f'rolling_max_{window}'] = df_engineered[target_col].rolling(window=window, min_periods=1).max()
    
    # 6. DIFFERENCE FEATURES
    df_engineered['diff_1'] = df_engineered[target_col].diff(1)
    df_engineered['diff_7'] = df_engineered[target_col].diff(7)
    
    # 7. HOLIDAY/SEASON INDICATORS (Simplified)
    # Summer months (AC usage)
    df_engineered['is_summer'] = df_engineered['month'].isin([3, 4, 5, 6]).astype(int)
    # Winter months
    df_engineered['is_winter'] = df_engineered['month'].isin([11, 12, 1, 2]).astype(int)
    # Festival season (Oct-Dec in India)
    df_engineered['is_festive'] = df_engineered['month'].isin([10, 11, 12]).astype(int)
    
    # 8. INTERACTION FEATURES
    if 'Temperature_C' in df_engineered.columns:
        df_engineered['temp_month_interaction'] = df_engineered['Temperature_C'] * df_engineered['month']
        df_engineered['temp_season_interaction'] = df_engineered['Temperature_C'] * df_engineered['is_summer']
    
    # 9. POLYNOMIAL TREND (quadratic and cubic)
    df_engineered['time_index_squared'] = df_engineered['time_index'] ** 2
    df_engineered['time_index_cubic'] = df_engineered['time_index'] ** 3
    
    # Drop NaN values created by lags
    df_engineered = df_engineered.dropna().reset_index(drop=True)
    
    return df_engineered

def prepare_forecast_dates(last_date, forecast_months, freq='D'):
    """Generate future dates for forecasting"""
    
    if freq == 'D':
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_months * 30,  # Approx 30 days per month
            freq='D'
        )
    elif freq == 'M':
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_months,
            freq='M'
        )
    
    return future_dates

def create_future_features(last_row, future_dates, feature_columns):
    """Create feature matrix for future predictions"""
    
    future_df = pd.DataFrame({'Date': future_dates})
    
    # Recreate all time-based features for future dates
    future_df['year'] = future_df['Date'].dt.year
    future_df['month'] = future_df['Date'].dt.month
    future_df['day'] = future_df['Date'].dt.day
    future_df['day_of_week'] = future_df['Date'].dt.dayofweek
    future_df['day_of_year'] = future_df['Date'].dt.dayofyear
    future_df['week_of_year'] = future_df['Date'].dt.isocalendar().week
    future_df['quarter'] = future_df['Date'].dt.quarter
    future_df['is_weekend'] = (future_df['day_of_week'] >= 5).astype(int)
    
    # Cyclical features
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
    future_df['week_sin'] = np.sin(2 * np.pi * future_df['week_of_year'] / 52)
    future_df['week_cos'] = np.cos(2 * np.pi * future_df['week_of_year'] / 52)
    future_df['day_sin'] = np.sin(2 * np.pi * future_df['day_of_year'] / 365.25)
    future_df['day_cos'] = np.cos(2 * np.pi * future_df['day_of_year'] / 365.25)
    future_df['dow_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
    future_df['dow_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
    
    # Time index (continue from last)
    time_start = last_row['time_index'] if 'time_index' in last_row else len(future_dates)
    future_df['time_index'] = np.arange(time_start + 1, time_start + 1 + len(future_dates))
    future_df['time_index_squared'] = future_df['time_index'] ** 2
    future_df['time_index_cubic'] = future_df['time_index'] ** 3
    
    # Seasonal indicators
    future_df['is_summer'] = future_df['month'].isin([3, 4, 5, 6]).astype(int)
    future_df['is_winter'] = future_df['month'].isin([11, 12, 1, 2]).astype(int)
    future_df['is_festive'] = future_df['month'].isin([10, 11, 12]).astype(int)
    
    # For lag/rolling features - we'll need to recursively predict
    # This is handled separately in the forecasting loop
    
    # Ensure we have all required columns
    for col in feature_columns:
        if col not in future_df.columns and col != 'Energy_Consumption_kWh':
            # Fill with median or appropriate value
            if col in last_row:
                future_df[col] = last_row[col]
            else:
                future_df[col] = 0
    
    return future_df

# ========== ML MODELS ==========
class EnergyForecaster:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def train_ml_models(self, X_train, y_train, X_test, y_test, features):
        """Train multiple ML models with time series features"""
        
        results = {}
        
        # Define models
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
        }
        
        for name, model in models_to_train.items():
            try:
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                metrics = {
                    'model': model,
                    'train_mae': mean_absolute_error(y_train, y_train_pred),
                    'test_mae': mean_absolute_error(y_test, y_test_pred),
                    'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                    'train_r2': r2_score(y_train, y_train_pred),
                    'test_r2': r2_score(y_test, y_test_pred),
                    'predictions': {
                        'train': y_train_pred,
                        'test': y_test_pred
                    }
                }
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    metrics['feature_importance'] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    metrics['feature_importance'] = np.abs(model.coef_)
                
                results[name] = metrics
                
            except Exception as e:
                st.warning(f"Error training {name}: {str(e)[:100]}")
                continue
        
        return results
    
    def train_arima(self, data, forecast_steps):
        """Train ARIMA model (traditional time series)"""
        try:
            # Simple ARIMA model
            model = ARIMA(data, order=(2, 1, 2))  # (p,d,q)
            model_fit = model.fit()
            
            # Forecast
            forecast = model_fit.forecast(steps=forecast_steps)
            
            return {
                'model': model_fit,
                'forecast': forecast,
                'aic': model_fit.aic,
                'bic': model_fit.bic
            }
        except Exception as e:
            st.warning(f"ARIMA Error: {str(e)}")
            return None
    
    def forecast_future_ml(self, model, last_row, future_dates, feature_columns, target_history):
        """Forecast future using ML model with recursive prediction for lags"""
        
        # Create base future features
        future_df = create_future_features(last_row, future_dates, feature_columns)
        
        # Initialize predictions list
        predictions = []
        history = list(target_history[-30:])  # Last 30 days as initial history
        
        # For each future date, predict recursively
        for i in range(len(future_dates)):
            # Update lag features based on previous predictions
            if i >= 1:
                history.append(predictions[-1])
            
            # Update rolling statistics
            if len(history) >= 7:
                future_df.loc[i, 'lag_1'] = history[-1] if i >= 1 else last_row['Energy_Consumption_kWh']
                future_df.loc[i, 'lag_7'] = history[-7] if len(history) >= 7 else np.mean(history)
                future_df.loc[i, 'rolling_mean_7'] = np.mean(history[-7:])
                future_df.loc[i, 'rolling_mean_30'] = np.mean(history[-30:]) if len(history) >= 30 else np.mean(history)
            
            # Prepare features for this date
            X_future = future_df.iloc[[i]][feature_columns].fillna(0)
            
            # Scale and predict
            X_future_scaled = self.scaler.transform(X_future)
            pred = model.predict(X_future_scaled)[0]
            predictions.append(max(0, pred))  # Ensure non-negative
        
        return predictions

# ========== MAIN APP ==========
def main():
    st.title("🔮 Household Electricity Consumption Forecast")
    st.markdown("### Predict future energy usage with machine learning")
    
    # Check if data is loaded
    if 'forecast_data' not in st.session_state or st.session_state.forecast_data is None:
        st.error("⚠️ No data loaded! Please go to Data Loader page first.")
        if st.button("📊 Go to Data Loader", type="primary"):
            st.switch_page("pages/2_Data_Loader.py")
        return
    
    # Load data
    data = st.session_state.forecast_data.copy()
    st.session_state.raw_data = data.copy()  # Store original
    
    # Display data info
    st.success(f"✅ Data loaded: {len(data)} records from {data['Date'].min().date()} to {data['Date'].max().date()}")
    
    # ========== FORECAST SETTINGS ==========
    st.markdown("---")
    st.markdown("### 📅 Forecast Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_months = st.number_input(
            "Months to Forecast",
            min_value=1,
            max_value=36,
            value=st.session_state.get('prediction_months', 12),
            help="How many months into the future to predict"
        )
    
    with col2:
        forecast_freq = st.selectbox(
            "Forecast Frequency",
            options=['Daily', 'Monthly'],
            index=0,
            help="Daily predictions are more detailed"
        )
        freq_code = 'D' if forecast_freq == 'Daily' else 'M'
    
    with col3:
        test_size = st.slider(
            "Test Data Percentage",
            min_value=10,
            max_value=40,
            value=20,
            help="Percentage of data to use for testing models"
        )
    
    # ========== DATA PREPARATION ==========
    st.markdown("---")
    st.markdown("### ⚙️ Data Preparation")
    
    # Engineer features
    with st.spinner("Engineering time-based features..."):
        data_engineered = engineer_time_features(data)
        
        # Display engineered features
        with st.expander("🔍 View Engineered Features", expanded=False):
            st.write(f"Original features: {list(data.columns)}")
            st.write(f"Engineered features: {list(data_engineered.columns)}")
            st.write(f"Total features: {len(data_engineered.columns)}")
            st.dataframe(data_engineered.head(), use_container_width=True)
    
    # Prepare features and target
    exclude_cols = ['Date', 'Energy_Consumption_kWh', 'Source', 'Location', 'Household_Size']
    feature_cols = [col for col in data_engineered.columns 
                   if col not in exclude_cols and data_engineered[col].dtype in ['int64', 'float64']]
    
    X = data_engineered[feature_cols].values
    y = data_engineered['Energy_Consumption_kWh'].values
    
    # Split data (time series aware)
    split_idx = int(len(X) * (1 - test_size/100))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # ========== MODEL TRAINING ==========
    st.markdown("---")
    st.markdown("### 🤖 Model Training & Comparison")
    
    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        
        with st.spinner("Training machine learning models..."):
            # Initialize forecaster
            forecaster = EnergyForecaster()
            
            # Train ML models
            ml_results = forecaster.train_ml_models(
                X_train, y_train, X_test, y_test, feature_cols
            )
            
            # Store results
            st.session_state.ml_results = ml_results
            st.session_state.forecaster = forecaster
            st.session_state.feature_cols = feature_cols
            st.session_state.data_engineered = data_engineered
            st.session_state.y_test = y_test
            st.session_state.test_dates = data_engineered['Date'].iloc[split_idx:].values
            
            st.success(f"✅ Trained {len(ml_results)} models successfully!")
    
    # ========== DISPLAY RESULTS ==========
    if 'ml_results' in st.session_state and st.session_state.ml_results:
        results = st.session_state.ml_results
        
        # Model Comparison Metrics
        st.markdown("---")
        st.markdown("### 📊 Model Performance Comparison")
        
        # Create metrics table
        metrics_data = []
        for name, res in results.items():
            metrics_data.append({
                'Model': name,
                'Test R²': res['test_r2'],
                'Test RMSE': res['test_rmse'],
                'Test MAE': res['test_mae'],
                'Train R²': res['train_r2'],
                'Overfit Δ': res['train_r2'] - res['test_r2'],
                'Status': '✅ Excellent' if res['test_r2'] > 0.8 else 
                         '🟡 Good' if res['test_r2'] > 0.6 else 
                         '🟠 Moderate' if res['test_r2'] > 0.4 else 
                         '🔴 Poor'
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics.style.format({
            'Test R²': '{:.3f}',
            'Test RMSE': '{:.2f}',
            'Test MAE': '{:.2f}',
            'Train R²': '{:.3f}',
            'Overfit Δ': '{:.3f}'
        }), use_container_width=True)
        
        # Best model
        best_model = max(results.items(), key=lambda x: x[1]['test_r2'])
        st.info(f"🏆 **Best Model**: {best_model[0]} with R² = {best_model[1]['test_r2']:.3f}")
        
        # ========== INDIVIDUAL MODEL VISUALIZATIONS ==========
        st.markdown("---")
        st.markdown("### 📈 Individual Model Analysis")
        
        # Model selector
        selected_model = st.selectbox(
            "Select Model to Visualize",
            list(results.keys()),
            key="model_selector"
        )
        
        if selected_model:
            model_res = results[selected_model]
            
            # Create tabs for this model
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictions", "📈 Residuals", "🎯 Feature Importance", "🔮 Future Forecast"])
            
            with tab1:
                # Plot predictions vs actual
                fig = go.Figure()
                
                # Test data
                fig.add_trace(go.Scatter(
                    x=st.session_state.test_dates,
                    y=st.session_state.y_test,
                    mode='lines',
                    name='Actual (Test)',
                    line=dict(color='blue', width=2),
                    opacity=0.7
                ))
                
                fig.add_trace(go.Scatter(
                    x=st.session_state.test_dates,
                    y=model_res['predictions']['test'],
                    mode='lines',
                    name=f'Predicted ({selected_model})',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f'{selected_model} - Predictions vs Actual',
                    xaxis_title='Date',
                    yaxis_title='Energy Consumption (kWh)',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics card
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    r2_color = "green" if model_res['test_r2'] > 0.7 else "orange" if model_res['test_r2'] > 0.5 else "red"
                    st.metric("R² Score", f"{model_res['test_r2']:.3f}", delta_color="off")
                
                with col2:
                    st.metric("RMSE", f"{model_res['test_rmse']:.2f}")
                
                with col3:
                    st.metric("MAE", f"{model_res['test_mae']:.2f}")
                
                with col4:
                    overfit = model_res['train_r2'] - model_res['test_r2']
                    st.metric("Overfitting", f"{overfit:.3f}", 
                             delta="Low" if overfit < 0.1 else "High" if overfit > 0.2 else "Moderate")
            
            with tab2:
                # Residual analysis
                residuals = st.session_state.y_test - model_res['predictions']['test']
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    # Residual histogram
                    fig_hist = px.histogram(
                        x=residuals,
                        nbins=30,
                        title='Residual Distribution',
                        labels={'x': 'Residuals (Actual - Predicted)'}
                    )
                    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col_res2:
                    # Residual vs Predicted
                    fig_scatter = px.scatter(
                        x=model_res['predictions']['test'],
                        y=residuals,
                        title='Residuals vs Predicted Values',
                        labels={'x': 'Predicted Values', 'y': 'Residuals'}
                    )
                    fig_scatter.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            with tab3:
                # Feature importance
                if 'feature_importance' in model_res:
                    importance = model_res['feature_importance']
                    
                    if len(importance) == len(st.session_state.feature_cols):
                        fi_df = pd.DataFrame({
                            'Feature': st.session_state.feature_cols,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False).head(15)
                        
                        fig = px.bar(
                            fi_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Top 15 Feature Importances',
                            color='Importance',
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.write("**Top Features:**")
                        for idx, row in fi_df.head(10).iterrows():
                            st.write(f"{row['Feature']}: {row['Importance']:.4f}")
                    else:
                        st.info("Feature importance not available for this model")
                else:
                    st.info("Feature importance not available for this model")
            
            with tab4:
                # FUTURE FORECASTING
                st.markdown("#### 🔮 Generate Future Forecast")
                
                if st.button("Generate Forecast", key=f"forecast_{selected_model}"):
                    with st.spinner(f"Generating {forecast_months}-month forecast..."):
                        forecaster = st.session_state.forecaster
                        model = model_res['model']
                        
                        # Get last data point
                        last_row = st.session_state.data_engineered.iloc[-1].to_dict()
                        
                        # Generate future dates
                        last_date = data['Date'].max()
                        future_dates = prepare_forecast_dates(last_date, forecast_months, freq_code)
                        
                        # Get history for recursive prediction
                        target_history = st.session_state.data_engineered['Energy_Consumption_kWh'].values
                        
                        # Generate forecast
                        future_predictions = forecaster.forecast_future_ml(
                            model, last_row, future_dates, 
                            st.session_state.feature_cols, target_history
                        )
                        
                        # Create forecast dataframe
                        forecast_df = pd.DataFrame({
                            'Date': future_dates,
                            'Predicted_Consumption_kWh': future_predictions,
                            'Model': selected_model
                        })
                        
                        # Store forecast
                        st.session_state.current_forecast = forecast_df
                        st.session_state.forecast_model = selected_model
                        
                        # Plot forecast
                        fig = go.Figure()
                        
                        # Historical data (last 90 days)
                        hist_days = min(90, len(data))
                        hist_data = data.iloc[-hist_days:]
                        
                        fig.add_trace(go.Scatter(
                            x=hist_data['Date'],
                            y=hist_data['Energy_Consumption_kWh'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Predicted_Consumption_kWh'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red', width=3)
                        ))
                        
                        # Confidence interval (simplified)
                        std_dev = np.std(forecast_df['Predicted_Consumption_kWh']) * 1.96
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
                            y=(forecast_df['Predicted_Consumption_kWh'] + std_dev).tolist() + 
                              (forecast_df['Predicted_Consumption_kWh'] - std_dev).tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='95% Confidence'
                        ))
                        
                        fig.update_layout(
                            title=f'{forecast_months}-Month Energy Forecast ({selected_model})',
                            xaxis_title='Date',
                            yaxis_title='Energy Consumption (kWh)',
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast metrics
                        st.markdown("#### 📊 Forecast Summary")
                        
                        col_f1, col_f2, col_f3 = st.columns(3)
                        
                        with col_f1:
                            avg_forecast = forecast_df['Predicted_Consumption_kWh'].mean()
                            st.metric("Average Forecast", f"{avg_forecast:.1f} kWh/day")
                        
                        with col_f2:
                            peak_forecast = forecast_df['Predicted_Consumption_kWh'].max()
                            st.metric("Peak Consumption", f"{peak_forecast:.1f} kWh")
                        
                        with col_f3:
                            total_forecast = forecast_df['Predicted_Consumption_kWh'].sum()
                            monthly_cost = (total_forecast / forecast_months) * 8  # ₹8 per kWh
                            st.metric("Est. Monthly Cost", f"₹{monthly_cost:.0f}")
                        
                        # Show forecast data
                        with st.expander("📋 View Forecast Data"):
                            st.dataframe(forecast_df, use_container_width=True)
                        
                        # Download forecast
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Forecast",
                            data=csv,
                            file_name=f"energy_forecast_{selected_model}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
        
        # ========== R² IMPROVEMENT TIPS ==========
        st.markdown("---")
        st.markdown("### 💡 How to Improve Low R² Scores")
        
        # Check if any model has poor R²
        poor_models = [name for name, res in results.items() if res['test_r2'] < 0.5]
        
        if poor_models:
            st.warning(f"⚠️ Low R² scores detected in: {', '.join(poor_models)}")
            
            col_tip1, col_tip2 = st.columns(2)
            
            with col_tip1:
                st.markdown("""
                #### 🔧 Technical Solutions:
                1. **Add External Data**
                   - Weather data (temperature, humidity)
                   - Holiday calendar
                   - Economic indicators
                
                2. **Better Feature Engineering**
                   - Add Fourier terms for seasonality
                   - Include interaction terms
                   - Create more lag features (up to 365 days)
                
                3. **Advanced Models**
                   - Try LSTM/GRU for sequential patterns
                   - Use Prophet (Facebook's time series model)
                   - Ensemble multiple models
                """)
            
            with col_tip2:
                st.markdown("""
                #### 📈 Data Quality:
                1. **More Historical Data**
                   - Aim for 2+ years of daily data
                   - Include multiple seasons
                
                2. **Data Cleaning**
                   - Handle outliers properly
                   - Fill missing values intelligently
                   - Remove anomalies
                
                3. **Granularity**
                   - Use hourly data instead of daily
                   - Include time-of-day patterns
                
                #### 🎯 Expected Improvements:
                - **More data**: +0.10 to +0.30 R²
                - **Better features**: +0.15 to +0.25 R²  
                - **Advanced models**: +0.10 to +0.20 R²
                """)
        
        # ========== SEASONALITY ANALYSIS ==========
        st.markdown("---")
        st.markdown("### 📅 Seasonality & Trend Analysis")
        
        if len(data) > 30:
            # Decompose time series
            try:
                data_ts = data.set_index('Date')['Energy_Consumption_kWh']
                decomposition = seasonal_decompose(data_ts, model='additive', period=30)
                
                fig_decompose = go.Figure()
                
                fig_decompose.add_trace(go.Scatter(
                    x=data_ts.index,
                    y=decomposition.trend,
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', width=2)
                ))
                
                fig_decompose.add_trace(go.Scatter(
                    x=data_ts.index,
                    y=decomposition.seasonal,
                    mode='lines',
                    name='Seasonal',
                    line=dict(color='green', width=1)
                ))
                
                fig_decompose.add_trace(go.Scatter(
                    x=data_ts.index,
                    y=decomposition.resid,
                    mode='lines',
                    name='Residual',
                    line=dict(color='blue', width=1, dash='dot')
                ))
                
                fig_decompose.update_layout(
                    title='Time Series Decomposition (Trend + Seasonal + Residual)',
                    xaxis_title='Date',
                    yaxis_title='Energy Consumption (kWh)',
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig_decompose, use_container_width=True)
                
                # Interpretation
                trend_strength = np.std(decomposition.trend.dropna()) / np.std(data_ts)
                seasonal_strength = np.std(decomposition.seasonal.dropna()) / np.std(data_ts)
                
                col_int1, col_int2 = st.columns(2)
                with col_int1:
                    st.metric("Trend Strength", f"{trend_strength:.2%}")
                with col_int2:
                    st.metric("Seasonal Strength", f"{seasonal_strength:.2%}")
                    
            except Exception as e:
                st.info(f"Seasonal decomposition requires at least 2 periods of data. {str(e)}")
    
    else:
        st.info("👆 Click 'Train All Models' to start forecasting analysis")
    
    # ========== NAVIGATION ==========
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    
    with col_nav1:
        if st.button("📊 Back to Data Loader", use_container_width=True):
            st.switch_page("pages/data_loader.py")
    
    with col_nav2:
        if st.button("📋 Back to Survey", use_container_width=True):
            st.switch_page("pages/survey.py")
    
    with col_nav3:
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            st.switch_page("main.py")

if __name__ == "__main__":
    main()
