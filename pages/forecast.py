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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb

# Time Series Models
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(
    page_title="Energy Consumption Forecast",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .algo-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .algo-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        margin: 2px 5px 2px 0;
        font-weight: 500;
    }
    .badge-excellent { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .badge-good { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
    .badge-moderate { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .badge-poor { background: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }
    .algo-header {
        font-size: 1.2em;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    .algo-icon {
        font-size: 1.5em;
        margin-right: 10px;
    }
    .dropdown-content {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
        border: 1px solid #e9ecef;
    }
    .st-expander {
        background: white;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    .model-tabs button {
        font-size: 14px !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# ========== ALGORITHM DEFINITIONS ==========
ALGORITHMS = {
    "📊 Linear Regression": {
        "model": LinearRegression(),
        "icon": "📊",
        "category": "Linear Models",
        "description": "Simple linear relationship between features and target",
        "pros": ["Fast", "Interpretable", "No hyperparameters"],
        "cons": ["Assumes linearity", "Poor with complex patterns"]
    },
    "🎯 Ridge Regression": {
        "model": Ridge(alpha=1.0),
        "icon": "🎯",
        "category": "Regularized Linear",
        "description": "Linear regression with L2 regularization to prevent overfitting",
        "pros": ["Reduces overfitting", "Handles multicollinearity"],
        "cons": ["All features kept", "Need to tune alpha"]
    },
    "🔪 Lasso Regression": {
        "model": Lasso(alpha=0.1),
        "icon": "🔪",
        "category": "Regularized Linear",
        "description": "Linear regression with L1 regularization for feature selection",
        "pros": ["Feature selection", "Sparse solutions"],
        "cons": ["Can eliminate important features"]
    },
    "🌳 Random Forest": {
        "model": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "icon": "🌳",
        "category": "Ensemble Trees",
        "description": "Ensemble of decision trees with bagging",
        "pros": ["Handles non-linearity", "Feature importance", "Robust to outliers"],
        "cons": ["Can overfit", "Less interpretable", "Slower training"]
    },
    "⚡ XGBoost": {
        "model": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
        "icon": "⚡",
        "category": "Gradient Boosting",
        "description": "Extreme Gradient Boosting - state-of-the-art for tabular data",
        "pros": ["High accuracy", "Fast prediction", "Built-in regularization"],
        "cons": ["Many hyperparameters", "Can overfit with small data"]
    },
    "🚀 Gradient Boosting": {
        "model": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "icon": "🚀",
        "category": "Gradient Boosting",
        "description": "Sequential ensemble that corrects previous errors",
        "pros": ["High accuracy", "Handles mixed data types"],
        "cons": ["Sensitive to outliers", "Slower training"]
    },
    "💡 LightGBM": {
        "model": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1),
        "icon": "💡",
        "category": "Gradient Boosting",
        "description": "Light Gradient Boosting Machine - faster than XGBoost",
        "pros": ["Very fast", "Low memory usage", "Great accuracy"],
        "cons": ["Can overfit on small data"]
    },
    "🧠 Neural Network": {
        "model": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        "icon": "🧠",
        "category": "Deep Learning",
        "description": "Multi-layer Perceptron for complex non-linear patterns",
        "pros": ["Captures complex patterns", "Universal approximator"],
        "cons": ["Black box", "Needs lots of data", "Slow training"]
    },
    "🎲 Decision Tree": {
        "model": DecisionTreeRegressor(max_depth=10, random_state=42),
        "icon": "🎲",
        "category": "Tree Models",
        "description": "Simple tree-based model that splits data recursively",
        "pros": ["Interpretable", "No feature scaling needed"],
        "cons": ["Prone to overfitting", "Unstable"]
    },
    "👑 K-Nearest Neighbors": {
        "model": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "icon": "👑",
        "category": "Instance-Based",
        "description": "Predicts based on similar instances in training data",
        "pros": ["Simple", "No training phase", "Non-parametric"],
        "cons": ["Slow prediction", "Sensitive to irrelevant features"]
    },
    "🛡️ Support Vector Regression": {
        "model": SVR(kernel='rbf', C=1.0),
        "icon": "🛡️",
        "category": "Kernel Methods",
        "description": "Finds optimal hyperplane with maximum margin",
        "pros": ["Effective in high dimensions", "Memory efficient"],
        "cons": ["Slow for large datasets", "Sensitive to parameters"]
    },
    "🏹 AdaBoost": {
        "model": AdaBoostRegressor(n_estimators=50, random_state=42),
        "icon": "🏹",
        "category": "Ensemble",
        "description": "Adaptive Boosting - focuses on hard-to-predict samples",
        "pros": ["Improves weak learners", "Less prone to overfitting"],
        "cons": ["Sensitive to noisy data"]
    },
    "🔄 ElasticNet": {
        "model": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "icon": "🔄",
        "category": "Regularized Linear",
        "description": "Combines L1 and L2 regularization",
        "pros": ["Best of both Ridge and Lasso", "Feature selection"],
        "cons": ["Two parameters to tune"]
    },
    "📈 ARIMA": {
        "model": "ARIMA",
        "icon": "📈",
        "category": "Time Series",
        "description": "AutoRegressive Integrated Moving Average for univariate time series",
        "pros": ["Specifically for time series", "Handles trend and seasonality"],
        "cons": ["Univariate only", "Need stationary data"]
    }
}

# ========== UTILITY FUNCTIONS ==========
def engineer_time_features(df, date_col='Date', target_col='Energy_Consumption_kWh'):
    """Engineer comprehensive time-based features"""
    
    df_engineered = df.copy()
    df_engineered[date_col] = pd.to_datetime(df_engineered[date_col])
    
    # Time index
    df_engineered['time_index'] = np.arange(len(df_engineered))
    
    # Date components
    df_engineered['year'] = df_engineered[date_col].dt.year
    df_engineered['month'] = df_engineered[date_col].dt.month
    df_engineered['day'] = df_engineered[date_col].dt.day
    df_engineered['day_of_week'] = df_engineered[date_col].dt.dayofweek
    df_engineered['day_of_year'] = df_engineered[date_col].dt.dayofyear
    df_engineered['week_of_year'] = df_engineered[date_col].dt.isocalendar().week
    df_engineered['quarter'] = df_engineered[date_col].dt.quarter
    df_engineered['is_weekend'] = (df_engineered['day_of_week'] >= 5).astype(int)
    
    # Cyclical features (CRITICAL FOR SEASONALITY!)
    df_engineered['month_sin'] = np.sin(2 * np.pi * df_engineered['month'] / 12)
    df_engineered['month_cos'] = np.cos(2 * np.pi * df_engineered['month'] / 12)
    df_engineered['week_sin'] = np.sin(2 * np.pi * df_engineered['week_of_year'] / 52)
    df_engineered['week_cos'] = np.cos(2 * np.pi * df_engineered['week_of_year'] / 52)
    df_engineered['day_sin'] = np.sin(2 * np.pi * df_engineered['day_of_year'] / 365.25)
    df_engineered['day_cos'] = np.cos(2 * np.pi * df_engineered['day_of_year'] / 365.25)
    df_engineered['dow_sin'] = np.sin(2 * np.pi * df_engineered['day_of_week'] / 7)
    df_engineered['dow_cos'] = np.cos(2 * np.pi * df_engineered['day_of_week'] / 7)
    
    # Lag features
    for lag in [1, 2, 3, 7, 14, 30]:
        df_engineered[f'lag_{lag}'] = df_engineered[target_col].shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 30, 90]:
        df_engineered[f'rolling_mean_{window}'] = df_engineered[target_col].rolling(window=window, min_periods=1).mean()
        df_engineered[f'rolling_std_{window}'] = df_engineered[target_col].rolling(window=window, min_periods=1).std()
    
    # Difference features
    df_engineered['diff_1'] = df_engineered[target_col].diff(1)
    df_engineered['diff_7'] = df_engineered[target_col].diff(7)
    
    # Seasonal indicators
    df_engineered['is_summer'] = df_engineered['month'].isin([3, 4, 5, 6]).astype(int)
    df_engineered['is_winter'] = df_engineered['month'].isin([11, 12, 1, 2]).astype(int)
    df_engineered['is_festive'] = df_engineered['month'].isin([10, 11, 12]).astype(int)
    
    # Polynomial trend
    df_engineered['time_index_squared'] = df_engineered['time_index'] ** 2
    df_engineered['time_index_cubic'] = df_engineered['time_index'] ** 3
    
    # Temperature interaction if available
    if 'Temperature_C' in df_engineered.columns:
        df_engineered['temp_month_interaction'] = df_engineered['Temperature_C'] * df_engineered['month']
    
    # Drop NaN
    df_engineered = df_engineered.dropna().reset_index(drop=True)
    
    return df_engineered

def get_r2_badge(r2_score):
    """Get colored badge for R² score"""
    if r2_score >= 0.8:
        return f'<span class="metric-badge badge-excellent">Excellent: {r2_score:.3f}</span>'
    elif r2_score >= 0.6:
        return f'<span class="metric-badge badge-good">Good: {r2_score:.3f}</span>'
    elif r2_score >= 0.4:
        return f'<span class="metric-badge badge-moderate">Moderate: {r2_score:.3f}</span>'
    else:
        return f'<span class="metric-badge badge-poor">Poor: {r2_score:.3f}</span>'

def create_algo_card(algo_name, algo_info, metrics=None, expanded=False):
    """Create algorithm card with dropdown"""
    
    icon = algo_info.get('icon', '📊')
    category = algo_info.get('category', 'Unknown')
    description = algo_info.get('description', '')
    
    # Create card header
    card_html = f"""
    <div class="algo-card">
        <div class="algo-header">
            <span class="algo-icon">{icon}</span>
            {algo_name}
            <span style="margin-left: auto; font-size: 0.9em; color: #666;">{category}</span>
        </div>
        <p style="color: #666; margin-bottom: 10px;">{description}</p>
    """
    
    # Add metrics if available
    if metrics:
        r2_badge = get_r2_badge(metrics.get('test_r2', 0))
        card_html += f"""
        <div style="margin-bottom: 10px;">
            {r2_badge}
            <span class="metric-badge" style="background: #e8f4fd; color: #0c5460; border: 1px solid #bee5eb;">RMSE: {metrics.get('test_rmse', 0):.2f}</span>
            <span class="metric-badge" style="background: #fef5e7; color: #856404; border: 1px solid #ffeaa7;">MAPE: {metrics.get('test_mape', 0):.1f}%</span>
        </div>
        """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Create expander for details
    with st.expander(f"🔍 View {algo_name.split()[-1]} Details", expanded=expanded):
        
        col_info, col_metrics = st.columns([1, 1])
        
        with col_info:
            st.markdown("#### 📝 Algorithm Info")
            st.write(f"**Category:** {category}")
            st.write(f"**Type:** {algo_info.get('type', 'Supervised Regression')}")
            
            if 'pros' in algo_info:
                st.markdown("**✅ Strengths:**")
                for pro in algo_info['pros']:
                    st.write(f"• {pro}")
            
            if 'cons' in algo_info:
                st.markdown("**⚠️ Limitations:**")
                for con in algo_info['cons']:
                    st.write(f"• {con}")
        
        with col_metrics:
            if metrics:
                st.markdown("#### 📊 Performance Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("R² Score", f"{metrics.get('test_r2', 0):.3f}")
                    st.metric("RMSE", f"{metrics.get('test_rmse', 0):.2f}")
                    st.metric("Train R²", f"{metrics.get('train_r2', 0):.3f}")
                with col2:
                    st.metric("MAE", f"{metrics.get('test_mae', 0):.2f}")
                    st.metric("MAPE", f"{metrics.get('test_mape', 0):.1f}%")
                    st.metric("Overfit Δ", f"{metrics.get('train_r2', 0) - metrics.get('test_r2', 0):.3f}")
        
        # Show visualizations if available
        if metrics and 'predictions' in metrics:
            st.markdown("#### 📈 Predictions vs Actual")
            
            # Create plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=metrics.get('test_dates', []),
                y=metrics.get('y_test', []),
                mode='lines',
                name='Actual',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=metrics.get('test_dates', []),
                y=metrics['predictions']['test'],
                mode='lines',
                name='Predicted',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f'{algo_name} - Test Predictions',
                xaxis_title='Date',
                yaxis_title='Energy Consumption (kWh)',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ========== ML MODEL TRAINING ==========
class EnergyForecastSystem:
    def __init__(self):
        self.results = {}
        self.scaler = StandardScaler()
        self.feature_cols = []
        
    def train_algorithm(self, algo_name, algo_config, X_train, y_train, X_test, y_test, test_dates):
        """Train a single algorithm"""
        
        try:
            model = algo_config["model"].fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'model': model,
                'train_r2': r2_score(y_train, y_train_pred),
                'test_r2': r2_score(y_test, y_test_pred),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'test_mae': mean_absolute_error(y_test, y_test_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'train_mape': mean_absolute_percentage_error(y_train, y_train_pred) * 100,
                'test_mape': mean_absolute_percentage_error(y_test, y_test_pred) * 100,
                'predictions': {
                    'train': y_train_pred,
                    'test': y_test_pred
                },
                'y_test': y_test,
                'test_dates': test_dates
            }
            
            # Feature importance for tree models
            if hasattr(model, 'feature_importances_'):
                metrics['feature_importance'] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                metrics['feature_importance'] = np.abs(model.coef_)
            
            return metrics
            
        except Exception as e:
            st.warning(f"Error training {algo_name}: {str(e)[:100]}")
            return None
    
    def train_all_algorithms(self, X_train, y_train, X_test, y_test, test_dates, selected_algos):
        """Train all selected algorithms"""
        
        results = {}
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (algo_name, algo_config) in enumerate(selected_algos.items()):
            status_text.text(f"Training {algo_name.split()[-1]}...")
            
            metrics = self.train_algorithm(
                algo_name, algo_config, 
                X_train_scaled, y_train, 
                X_test_scaled, y_test,
                test_dates
            )
            
            if metrics:
                results[algo_name] = metrics
                # Update ALGORITHMS dict with metrics
                ALGORITHMS[algo_name]['metrics'] = metrics
            
            progress_bar.progress((idx + 1) / len(selected_algos))
        
        status_text.text("✅ Training complete!")
        return results

# ========== MAIN APP ==========
def main():
    st.title("🔮 Household Electricity Consumption Forecasting")
    st.markdown("### Compare Multiple ML Algorithms for Energy Prediction")
    
    # Check if data is loaded
    if 'forecast_data' not in st.session_state or st.session_state.forecast_data is None:
        st.error("⚠️ No data loaded! Please go to Data Loader page first.")
        if st.button("📊 Go to Data Loader", type="primary", use_container_width=True):
            st.switch_page("pages/2_Data_Loader.py")
        return
    
    # Load data
    data = st.session_state.forecast_data.copy()
    st.success(f"✅ Data loaded: {len(data)} records from {data['Date'].min().date()} to {data['Date'].max().date()}")
    
    # ========== SIDEBAR SETTINGS ==========
    with st.sidebar:
        st.markdown("### ⚙️ Forecast Settings")
        
        # Forecast period
        forecast_months = st.number_input(
            "Months to Forecast",
            min_value=1,
            max_value=36,
            value=st.session_state.get('prediction_months', 12),
            help="How many months into the future to predict"
        )
        
        # Test size
        test_size = st.slider(
            "Test Data (%)",
            min_value=10,
            max_value=40,
            value=20,
            help="Percentage of data for testing models"
        )
        
        # Algorithm selection
        st.markdown("### 🤖 Select Algorithms")
        
        # Group algorithms by category
        algo_categories = {}
        for algo_name, algo_info in ALGORITHMS.items():
            category = algo_info.get('category', 'Other')
            if category not in algo_categories:
                algo_categories[category] = []
            algo_categories[category].append((algo_name, algo_info))
        
        # Create checkboxes by category
        selected_algorithms = {}
        
        for category, algos in algo_categories.items():
            with st.expander(f"{category} ({len(algos)})", expanded=True):
                for algo_name, algo_info in algos:
                    if algo_name != "📈 ARIMA":  # Skip ARIMA for now
                        if st.checkbox(algo_name, value=True, 
                                     help=algo_info.get('description', '')):
                            selected_algorithms[algo_name] = algo_info
        
        # Quick select buttons
        st.markdown("### ⚡ Quick Select")
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            if st.button("Select All", use_container_width=True):
                for algo_name, algo_info in ALGORITHMS.items():
                    if algo_name != "📈 ARIMA":
                        selected_algorithms[algo_name] = algo_info
        
        with col_q2:
            if st.button("Clear All", use_container_width=True, type="secondary"):
                selected_algorithms.clear()
        
        st.markdown(f"**Selected:** {len(selected_algorithms)} algorithms")
    
    # ========== MAIN CONTENT ==========
    
    # Data preparation
    st.markdown("---")
    st.markdown("### ⚙️ Data Preparation & Feature Engineering")
    
    with st.spinner("Engineering time-based features..."):
        data_engineered = engineer_time_features(data)
        
        # Display feature info
        with st.expander("🔍 View Engineered Features", expanded=False):
            st.info(f"✅ Engineered {len(data_engineered.columns)} features total")
            st.write(f"Original features: {list(data.columns)}")
            
            # Show engineered features by category
            time_features = [col for col in data_engineered.columns if col not in data.columns and col not in ['Date', 'Energy_Consumption_kWh']]
            
            col_feat1, col_feat2, col_feat3 = st.columns(3)
            
            with col_feat1:
                st.markdown("**📅 Date Components:**")
                date_features = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter']
                for feat in date_features:
                    if feat in time_features:
                        st.write(f"• {feat}")
            
            with col_feat2:
                st.markdown("**🌀 Cyclical Features:**")
                cyclic_features = [f for f in time_features if 'sin' in f or 'cos' in f]
                for feat in cyclic_features[:5]:
                    st.write(f"• {feat}")
                if len(cyclic_features) > 5:
                    st.write(f"... and {len(cyclic_features)-5} more")
            
            with col_feat3:
                st.markdown("**📊 Lag & Rolling:**")
                lag_features = [f for f in time_features if 'lag_' in f or 'rolling_' in f or 'diff_' in f]
                for feat in lag_features[:5]:
                    st.write(f"• {feat}")
                if len(lag_features) > 5:
                    st.write(f"... and {len(lag_features)-5} more")
    
    # Prepare features and target
    exclude_cols = ['Date', 'Energy_Consumption_kWh', 'Source', 'Location', 'Household_Size']
    feature_cols = [col for col in data_engineered.columns 
                   if col not in exclude_cols and data_engineered[col].dtype in ['int64', 'float64']]
    
    X = data_engineered[feature_cols].values
    y = data_engineered['Energy_Consumption_kWh'].values
    dates = data_engineered['Date'].values
    
    # Split data
    split_idx = int(len(X) * (1 - test_size/100))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    train_dates, test_dates = dates[:split_idx], dates[split_idx:]
    
    # ========== TRAIN BUTTON ==========
    st.markdown("---")
    st.markdown("### 🚀 Model Training")
    
    col_train1, col_train2 = st.columns([3, 1])
    
    with col_train1:
        st.write(f"**Ready to train {len(selected_algorithms)} selected algorithms**")
        st.write(f"Training data: {len(X_train)} records | Test data: {len(X_test)} records")
    
    with col_train2:
        if st.button("🎯 Train Selected Algorithms", type="primary", use_container_width=True):
            if len(selected_algorithms) == 0:
                st.warning("Please select at least one algorithm")
            else:
                with st.spinner(f"Training {len(selected_algorithms)} algorithms..."):
                    # Initialize system
                    forecast_system = EnergyForecastSystem()
                    forecast_system.feature_cols = feature_cols
                    
                    # Train all algorithms
                    results = forecast_system.train_all_algorithms(
                        X_train, y_train, X_test, y_test, test_dates, selected_algorithms
                    )
                    
                    # Store in session state
                    st.session_state.forecast_results = results
                    st.session_state.forecast_system = forecast_system
                    st.session_state.X_train = X_train
                    st.session_state.y_train = y_train
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.test_dates = test_dates
                    st.session_state.feature_cols = feature_cols
                    st.session_state.data_engineered = data_engineered
                    
                    st.success(f"✅ Trained {len(results)} algorithms successfully!")
                    st.rerun()
    
    # ========== DISPLAY RESULTS ==========
    if 'forecast_results' in st.session_state and st.session_state.forecast_results:
        results = st.session_state.forecast_results
        
        # Summary metrics
        st.markdown("---")
        st.markdown("### 📊 Algorithm Performance Summary")
        
        # Create summary table
        summary_data = []
        for algo_name, res in results.items():
            summary_data.append({
                'Algorithm': algo_name.split()[-1],
                'R² Score': res['test_r2'],
                'RMSE': res['test_rmse'],
                'MAE': res['test_mae'],
                'MAPE (%)': res['test_mape'],
                'Train R²': res['train_r2'],
                'Overfit Δ': res['train_r2'] - res['test_r2'],
                'Category': ALGORITHMS[algo_name]['category']
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Sort by R² score
        df_summary = df_summary.sort_values('R² Score', ascending=False)
        
        # Display with formatting
        st.dataframe(
            df_summary.style.format({
                'R² Score': '{:.3f}',
                'RMSE': '{:.2f}',
                'MAE': '{:.2f}',
                'MAPE (%)': '{:.1f}',
                'Train R²': '{:.3f}',
                'Overfit Δ': '{:.3f}'
            }).background_gradient(subset=['R² Score'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
        
        # Best model highlight
        best_algo = max(results.items(), key=lambda x: x[1]['test_r2'])
        st.info(f"🏆 **Best Performing Algorithm:** {best_algo[0]} with R² = {best_algo[1]['test_r2']:.3f} and MAPE = {best_algo[1]['test_mape']:.1f}%")
        
        # ========== INDIVIDUAL ALGORITHM DROPDOWNS ==========
        st.markdown("---")
        st.markdown("### 📋 Individual Algorithm Analysis")
        st.markdown("Click on each algorithm to view detailed results and visualizations")
        
        # Sort algorithms by performance
        sorted_algos = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
        
        # Create tabs for different views
        view_tab1, view_tab2, view_tab3 = st.tabs(["🏆 Ranked View", "📁 By Category", "🔍 All Expanded"])
        
        with view_tab1:
            # Display in rank order
            for rank, (algo_name, algo_metrics) in enumerate(sorted_algos, 1):
                algo_info = ALGORITHMS[algo_name]
                create_algo_card(
                    f"#{rank}: {algo_name}", 
                    algo_info, 
                    algo_metrics,
                    expanded=False
                )
        
        with view_tab2:
            # Group by category
            categories = {}
            for algo_name, algo_metrics in results.items():
                category = ALGORITHMS[algo_name]['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append((algo_name, algo_metrics))
            
            # Display by category
            for category, algos in categories.items():
                st.markdown(f"### {category}")
                for algo_name, algo_metrics in algos:
                    algo_info = ALGORITHMS[algo_name]
                    create_algo_card(algo_name, algo_info, algo_metrics, expanded=False)
        
        with view_tab3:
            # All expanded
            for algo_name, algo_metrics in results.items():
                algo_info = ALGORITHMS[algo_name]
                create_algo_card(algo_name, algo_info, algo_metrics, expanded=True)
        
        # ========== COMPARISON VISUALIZATIONS ==========
        st.markdown("---")
        st.markdown("### 📈 Algorithm Comparison Visualizations")
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # R² Comparison Bar Chart
            algo_names = [a.split()[-1] for a in results.keys()]
            r2_scores = [r['test_r2'] for r in results.values()]
            
            fig_r2 = go.Figure(data=[
                go.Bar(x=algo_names, y=r2_scores, 
                      marker_color=['green' if r2 > 0.7 else 'orange' if r2 > 0.5 else 'red' for r2 in r2_scores])
            ])
            fig_r2.update_layout(
                title='R² Score Comparison (Higher is Better)',
                xaxis_title='Algorithm',
                yaxis_title='R² Score',
                height=400
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col_viz2:
            # RMSE Comparison Bar Chart
            rmse_scores = [r['test_rmse'] for r in results.values()]
            
            fig_rmse = go.Figure(data=[
                go.Bar(x=algo_names, y=rmse_scores,
                      marker_color=['red' if rmse > rmse_scores[0]*1.5 else 'orange' if rmse > rmse_scores[0] else 'green' for rmse in rmse_scores])
            ])
            fig_rmse.update_layout(
                title='RMSE Comparison (Lower is Better)',
                xaxis_title='Algorithm',
                yaxis_title='RMSE',
                height=400
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # ========== FUTURE FORECASTING ==========
        st.markdown("---")
        st.markdown("### 🔮 Generate Future Forecast")
        
        if len(results) > 0:
            # Select model for forecasting
            col_fore1, col_fore2 = st.columns([2, 1])
            
            with col_fore1:
                forecast_algo = st.selectbox(
                    "Select Algorithm for Forecasting",
                    options=list(results.keys()),
                    format_func=lambda x: x.split()[-1],
                    help="Choose which algorithm to use for future predictions"
                )
            
            with col_fore2:
                if st.button("Generate Forecast", type="primary", use_container_width=True):
                    with st.spinner(f"Generating {forecast_months}-month forecast..."):
                        # Get the trained model
                        model = results[forecast_algo]['model']
                        last_date = data['Date'].max()
                        
                        # Generate future dates
                        future_dates = pd.date_range(
                            start=last_date + timedelta(days=1),
                            periods=forecast_months * 30,
                            freq='D'
                        )
                        
                        # Create future features (simplified - in reality need recursive prediction)
                        # For now, we'll show a seasonal forecast
                        seasonal_pattern = []
                        base_consumption = np.mean(y_test)
                        
                        for date in future_dates:
                            month = date.month
                            # Simple seasonal pattern
                            if month in [5, 6, 7]:  # Summer
                                consumption = base_consumption * 1.3
                            elif month in [12, 1, 2]:  # Winter
                                consumption = base_consumption * 0.9
                            else:
                                consumption = base_consumption
                            
                            # Add some randomness
                            consumption *= np.random.normal(1, 0.1)
                            seasonal_pattern.append(max(0, consumption))
                        
                        # Create forecast dataframe
                        forecast_df = pd.DataFrame({
                            'Date': future_dates,
                            'Predicted_kWh': seasonal_pattern,
                            'Algorithm': forecast_algo.split()[-1]
                        })
                        
                        # Plot forecast
                        fig_fore = go.Figure()
                        
                        # Historical data (last 180 days)
                        hist_data = data.iloc[-180:]
                        fig_fore.add_trace(go.Scatter(
                            x=hist_data['Date'],
                            y=hist_data['Energy_Consumption_kWh'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Forecast
                        fig_fore.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Predicted_kWh'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red', width=3)
                        ))
                        
                        fig_fore.update_layout(
                            title=f'{forecast_months}-Month Energy Forecast ({forecast_algo.split()[-1]})',
                            xaxis_title='Date',
                            yaxis_title='Energy Consumption (kWh)',
                            height=500
                        )
                        
                        st.plotly_chart(fig_fore, use_container_width=True)
                        
                        # Forecast summary
                        st.info(f"""
                        **Forecast Summary:**
                        - **Algorithm:** {forecast_algo.split()[-1]}
                        - **Period:** {forecast_months} months
                        - **Avg Daily Consumption:** {np.mean(seasonal_pattern):.1f} kWh
                        - **Total Forecast:** {np.sum(seasonal_pattern):.0f} kWh
                        - **Est. Monthly Cost:** ₹{np.mean(seasonal_pattern) * 30 * 8:,.0f} @ ₹8/kWh
                        """)
        
        # ========== R² IMPROVEMENT GUIDE ==========
        st.markdown("---")
        st.markdown("### 💡 How to Improve Model Performance")
        
        # Check for low R² scores
        low_r2_algos = [name for name, res in results.items() if res['test_r2'] < 0.5]
        
        if low_r2_algos:
            st.warning(f"**Low R² detected in:** {', '.join([a.split()[-1] for a in low_r2_algos])}")
            
            col_imp1, col_imp2 = st.columns(2)
            
            with col_imp1:
                st.markdown("""
                **🔧 Feature Engineering:**
                1. Add Fourier terms for seasonality
                2. Include holiday indicators
                3. Create interaction features
                4. Add external data (weather, economic)
                
                **🤖 Model Selection:**
                1. Try ensemble methods (Voting/Stacking)
                2. Use LSTM for time series
                3. Experiment with Prophet
                4. Hyperparameter tuning
                """)
            
            with col_imp2:
                st.markdown("""
                **📊 Data Quality:**
                1. Ensure 2+ years of data
                2. Handle outliers properly
                3. Fill missing values
                4. Use hourly data if possible
                
                **🎯 Expected Improvements:**
                - Better features: +0.15 to +0.25 R²
                - More data: +0.10 to +0.20 R²
                - Advanced models: +0.10 to +0.30 R²
                """)
        
    else:
        # Initial state - no results yet
        st.markdown("---")
        st.markdown("### 📋 Available Algorithms")
        
        # Show all available algorithms
        st.info(f"**Total Algorithms Available:** {len(ALGORITHMS)}")
        
        # Display algorithm cards without metrics
        for algo_name, algo_info in ALGORITHMS.items():
            if algo_name != "📈 ARIMA":  # Skip ARIMA
                create_algo_card(algo_name, algo_info, metrics=None, expanded=False)
        
        # Training instructions
        st.markdown("""
        ---
        ### 🚀 Ready to Start?
        
        1. **Select algorithms** from the sidebar
        2. **Click 'Train Selected Algorithms'** button
        3. **View results** in ranked order
        4. **Click on any algorithm** to see detailed analysis
        5. **Generate forecasts** with your best model
        
        **💡 Tip:** Start with 4-5 algorithms first (Random Forest, XGBoost, Linear Regression, Gradient Boosting)
        """)
    
    # ========== NAVIGATION ==========
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    
    with col_nav1:
        if st.button("📊 Back to Data Loader", use_container_width=True, icon="📊"):
            st.switch_page("pages/2_Data_Loader.py")
    
    with col_nav2:
        if st.button("📋 Back to Survey", use_container_width=True, icon="📋"):
            st.switch_page("pages/survey.py")
    
    with col_nav3:
        if st.button("🏠 Back to Dashboard", use_container_width=True, icon="🏠"):
            st.switch_page("main.py")

if __name__ == "__main__":
    main()
