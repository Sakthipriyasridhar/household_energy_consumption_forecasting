import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set page config at the very beginning
st.set_page_config(
    page_title="Household Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
    .stSelectbox, .stNumberInput, .stDateInput {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class EnergyForecastDashboard:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.df = None
        self.features = None
        
    def generate_sample_data(self, n_samples=10000):
        """Generate realistic household energy consumption data"""
        np.random.seed(42)
        
        # Create date range
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        
        # Create features specific to household energy consumption
        data = {
            'timestamp': dates,
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'day_of_month': dates.day,
            'month': dates.month,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'is_night': ((dates.hour >= 22) | (dates.hour <= 6)).astype(int),
            'is_peak_hours': ((dates.hour >= 17) & (dates.hour <= 21)).astype(int),
            
            # Household features
            'temperature': 15 + 15 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 3, n_samples),
            'humidity': 50 + 20 * np.sin(2 * np.pi * dates.hour / 12) + np.random.normal(0, 5, n_samples),
            'occupancy': np.random.poisson(2, n_samples) * ((dates.hour >= 7) & (dates.hour <= 22)),
            'appliance_usage': np.random.exponential(1, n_samples) * ((dates.hour >= 8) & (dates.hour <= 23)),
            
            # Time-based patterns
            'trend': np.arange(n_samples) * 0.002,
            'seasonality': 5 * np.sin(2 * np.pi * np.arange(n_samples) / 24) +  # Daily
                          3 * np.sin(2 * np.pi * np.arange(n_samples) / 168) +  # Weekly
                          2 * np.sin(2 * np.pi * np.arange(n_samples) / 720),   # Monthly
            
            # Random noise
            'noise': np.random.normal(0, 0.5, n_samples)
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate energy consumption based on features
        df['energy_consumption'] = (
            2.0 +  # Base consumption
            0.8 * df['temperature'] +
            0.3 * df['humidity'] +
            1.2 * df['occupancy'] +
            1.5 * df['appliance_usage'] +
            0.6 * df['trend'] +
            0.7 * df['seasonality'] +
            0.5 * df['is_weekend'] * df['occupancy'] +
            0.8 * df['is_peak_hours'] * df['appliance_usage'] +
            df['noise']
        )
        
        # Ensure no negative values
        df['energy_consumption'] = df['energy_consumption'].clip(lower=0.1)
        
        return df
    
    def engineer_features_fixed(self, df):
        """Fixed feature engineering function to avoid the error"""
        
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Create lag features safely
        for lag in [1, 24, 168]:  # 1 hour, 1 day, 1 week
            df_processed[f'energy_lag_{lag}'] = df_processed['energy_consumption'].shift(lag)
        
        # Create rolling statistics safely
        windows = [12, 24, 168]  # 12 hours, 1 day, 1 week
        
        for window in windows:
            df_processed[f'energy_rolling_mean_{window}'] = (
                df_processed['energy_consumption']
                .rolling(window=window, min_periods=1)
                .mean()
            )
            df_processed[f'energy_rolling_std_{window}'] = (
                df_processed['energy_consumption']
                .rolling(window=window, min_periods=1)
                .std()
            )
        
        # Create rolling statistics for temperature (FIXED - no typo)
        df_processed['temperature_rolling_mean_12'] = (
            df_processed['temperature']
            .rolling(window=12, min_periods=1)
            .mean()
        )
        df_processed['temperature_rolling_std_12'] = (
            df_processed['temperature']
            .rolling(window=12, min_periods=1)
            .std()
        )
        
        # Create interaction features
        df_processed['temp_hour_interaction'] = df_processed['temperature'] * df_processed['hour']
        df_processed['occupancy_temp_interaction'] = df_processed['occupancy'] * df_processed['temperature']
        df_processed['appliance_weekend_interaction'] = (
            df_processed['appliance_usage'] * df_processed['is_weekend']
        )
        
        # Add Fourier terms for seasonality
        for period in [24, 168, 720, 8760]:  # Daily, weekly, monthly, yearly
            df_processed[f'fourier_sin_{period}'] = np.sin(2 * np.pi * df_processed.index / period)
            df_processed[f'fourier_cos_{period}'] = np.cos(2 * np.pi * df_processed.index / period)
        
        # Add time-based features
        df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['hour'] / 24)
        df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['hour'] / 24)
        df_processed['day_sin'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
        df_processed['day_cos'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)
        
        # Target encoding for categorical features
        for col in ['hour', 'day_of_week']:
            mean_encoding = df_processed.groupby(col)['energy_consumption'].transform('mean')
            df_processed[f'{col}_energy_encoded'] = mean_encoding
        
        # Create difference features
        df_processed['energy_diff_1'] = df_processed['energy_consumption'].diff(1)
        df_processed['energy_diff_24'] = df_processed['energy_consumption'].diff(24)
        
        # Drop NaN values from lag and diff features
        df_processed = df_processed.dropna().reset_index(drop=True)
        
        # Select features for modeling
        feature_columns = [
            'hour', 'day_of_week', 'day_of_month', 'month',
            'is_weekend', 'is_night', 'is_peak_hours',
            'temperature', 'humidity', 'occupancy', 'appliance_usage',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        # Add all engineered features
        engineered_features = [
            col for col in df_processed.columns 
            if col not in ['timestamp', 'energy_consumption', 'trend', 'seasonality', 'noise']
            and col not in feature_columns
        ]
        
        all_features = feature_columns + engineered_features
        
        X = df_processed[all_features].values
        y = df_processed['energy_consumption'].values
        
        return X, y, all_features, df_processed
    
    def train_models(self, X, y, features):
        """Train multiple models with hyperparameter tuning"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models with simpler hyperparameters for Streamlit
        models_config = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Ridge Regression': {
                'model': Ridge(random_state=42),
                'params': {'alpha': [0.1, 1.0, 10.0]}
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1]
                }
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            }
        }
        
        results = {}
        
        # Progress tracking for Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (name, config) in enumerate(models_config.items()):
            status_text.text(f"Training {name}...")
            
            try:
                if config['params']:
                    # Grid search for models with hyperparameters
                    grid_search = GridSearchCV(
                        config['model'],
                        config['params'],
                        cv=TimeSeriesSplit(n_splits=3),
                        scoring='r2',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train_scaled, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                else:
                    # Simple fit for models without hyperparameters
                    best_model = config['model']
                    best_model.fit(X_train_scaled, y_train)
                    best_params = {}
                
                # Predictions
                y_pred_train = best_model.predict(X_train_scaled)
                y_pred_test = best_model.predict(X_test_scaled)
                
                # Calculate metrics
                metrics = {
                    'model': best_model,
                    'train_r2': r2_score(y_train, y_pred_train),
                    'test_r2': r2_score(y_test, y_pred_test),
                    'train_mape': mean_absolute_percentage_error(y_train, y_pred_train) * 100,
                    'test_mape': mean_absolute_percentage_error(y_test, y_pred_test) * 100,
                    'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'best_params': best_params,
                    'predictions': {
                        'train': y_pred_train,
                        'test': y_pred_test
                    }
                }
                
                # Feature importance if available
                if hasattr(best_model, 'feature_importances_'):
                    metrics['feature_importance'] = best_model.feature_importances_
                elif hasattr(best_model, 'coef_'):
                    metrics['feature_importance'] = np.abs(best_model.coef_)
                else:
                    metrics['feature_importance'] = None
                
                results[name] = metrics
                
            except Exception as e:
                st.warning(f"Error training {name}: {str(e)[:100]}...")
                continue
            
            # Update progress
            progress_bar.progress((idx + 1) / len(models_config))
        
        status_text.text("Training complete!")
        
        self.results = results
        return results, X_test, y_test
    
    def display_metrics_dashboard(self):
        """Display metrics dashboard in Streamlit"""
        
        st.markdown("<h2 class='main-header'>📊 Model Performance Dashboard</h2>", unsafe_allow_html=True)
        
        if not self.results:
            st.warning("No models trained yet. Please train models first.")
            return
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best_model = max(self.results.items(), key=lambda x: x[1]['test_r2'])
            st.metric(
                label="Best Model",
                value=best_model[0],
                delta=f"R²: {best_model[1]['test_r2']:.3f}"
            )
        
        with col2:
            avg_r2 = np.mean([m['test_r2'] for m in self.results.values()])
            st.metric(
                label="Average R²",
                value=f"{avg_r2:.3f}",
                delta="All Models"
            )
        
        with col3:
            best_r2 = max([m['test_r2'] for m in self.results.values()])
            st.metric(
                label="Best R² Score",
                value=f"{best_r2:.3f}",
                delta="Excellent" if best_r2 > 0.8 else "Good" if best_r2 > 0.6 else "Needs Improvement"
            )
        
        with col4:
            best_mape = min([m['test_mape'] for m in self.results.values()])
            st.metric(
                label="Best MAPE",
                value=f"{best_mape:.2f}%",
                delta="Lower is better"
            )
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Performance Comparison",
            "🔍 Individual Models",
            "🎯 Feature Importance",
            "📋 Detailed Metrics"
        ])
        
        with tab1:
            self.plot_performance_comparison()
        
        with tab2:
            self.plot_individual_models()
        
        with tab3:
            self.plot_feature_importance()
        
        with tab4:
            self.display_detailed_metrics()
    
    def plot_performance_comparison(self):
        """Plot model performance comparison"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        models = list(self.results.keys())
        
        # Plot 1: R² Scores
        train_r2 = [self.results[m]['train_r2'] for m in models]
        test_r2 = [self.results[m]['test_r2'] for m in models]
        
        x = np.arange(len(models))
        ax1.bar(x - 0.2, train_r2, 0.4, label='Train', color='lightblue', alpha=0.8)
        ax1.bar(x + 0.2, test_r2, 0.4, label='Test', color='steelblue', alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Train vs Test R² Scores')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MAPE Comparison
        test_mape = [self.results[m]['test_mape'] for m in models]
        colors = ['green' if m < 10 else 'orange' if m < 20 else 'red' for m in test_mape]
        bars = ax2.bar(models, test_mape, color=colors, alpha=0.8)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('MAPE (%)')
        ax2.set_title('Test MAPE Scores')
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, test_mape):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: RMSE Comparison
        test_rmse = [self.results[m]['test_rmse'] for m in models]
        ax3.bar(models, test_rmse, color='lightcoral', alpha=0.8)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('RMSE')
        ax3.set_title('Test RMSE Scores')
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Composite Score
        composite_scores = []
        for model in models:
            r2 = self.results[model]['test_r2']
            mape = self.results[model]['test_mape']
            # Composite score where higher is better
            composite = r2 * 0.7 + (1 / (1 + mape/100)) * 0.3
            composite_scores.append(composite)
        
        ax4.bar(models, composite_scores, color='lightgreen', alpha=0.8)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Composite Score')
        ax4.set_title('Model Ranking (Composite Score)')
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def plot_individual_models(self):
        """Plot individual model predictions"""
        
        selected_model = st.selectbox(
            "Select Model to View",
            list(self.results.keys()),
            key="model_selector"
        )
        
        if selected_model:
            metrics = self.results[selected_model]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Test R²", f"{metrics['test_r2']:.3f}")
                st.metric("Test MAPE", f"{metrics['test_mape']:.2f}%")
                st.metric("Test RMSE", f"{metrics['test_rmse']:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                # Create prediction plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot last 100 predictions
                y_pred = metrics['predictions']['test'][-100:]
                y_true = self.y_test[-100:] if hasattr(self, 'y_test') else None
                
                if y_true is not None:
                    ax.plot(y_true, label='Actual', color='blue', linewidth=2, alpha=0.7)
                    ax.plot(y_pred, label='Predicted', color='red', linewidth=1.5, linestyle='--', alpha=0.9)
                    ax.fill_between(range(len(y_pred)),
                                   y_pred - metrics['test_rmse'],
                                   y_pred + metrics['test_rmse'],
                                   alpha=0.2, color='gray', label='±RMSE')
                    
                    ax.set_xlabel('Time Steps')
                    ax.set_ylabel('Energy Consumption')
                    ax.set_title(f'{selected_model} - Predictions vs Actual')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        
        # Get models with feature importance
        models_with_fi = [
            m for m in self.results.keys() 
            if self.results[m]['feature_importance'] is not None
        ]
        
        if not models_with_fi:
            st.info("Feature importance not available for selected models.")
            return
        
        selected_model = st.selectbox(
            "Select Model for Feature Importance",
            models_with_fi,
            key="fi_selector"
        )
        
        if selected_model and self.features:
            importance = self.results[selected_model]['feature_importance']
            
            # Ensure we have the right number of features
            n_features = min(len(importance), len(self.features), 20)  # Show top 20
            
            # Get indices of top features
            top_idx = np.argsort(importance)[-n_features:]
            top_features = [self.features[i] for i in top_idx]
            top_importance = importance[top_idx]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(range(n_features), top_importance)
            ax.set_yticks(range(n_features))
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Importance Score')
            ax.set_title(f'{selected_model} - Top {n_features} Feature Importance')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Color bars by importance
            for bar, imp in zip(bars, top_importance):
                bar.set_color(plt.cm.viridis(imp / max(top_importance)))
            
            plt.tight_layout()
            st.pyplot(fig)
    
    def display_detailed_metrics(self):
        """Display detailed metrics table"""
        
        # Create DataFrame with all metrics
        metrics_data = []
        for model_name, metrics in self.results.items():
            metrics_data.append({
                'Model': model_name,
                'Train R²': f"{metrics['train_r2']:.4f}",
                'Test R²': f"{metrics['test_r2']:.4f}",
                'Train MAPE': f"{metrics['train_mape']:.2f}%",
                'Test MAPE': f"{metrics['test_mape']:.2f}%",
                'Train RMSE': f"{metrics['train_rmse']:.2f}",
                'Test RMSE': f"{metrics['test_rmse']:.2f}",
                'Overfitting': f"{(metrics['train_r2'] - metrics['test_r2']):.4f}",
                'Status': '✅ Good' if metrics['test_r2'] > 0.7 else 
                         '⚠️ Moderate' if metrics['test_r2'] > 0.5 else 
                         '❌ Needs Improvement'
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Display table
        st.dataframe(df_metrics, use_container_width=True)
        
        # Download button
        csv = df_metrics.to_csv(index=False)
        st.download_button(
            label="📥 Download Metrics as CSV",
            data=csv,
            file_name="model_metrics.csv",
            mime="text/csv"
        )

def main():
    """Main Streamlit app"""
    
    st.sidebar.markdown("<h2>⚡ Energy Forecast Dashboard</h2>", unsafe_allow_html=True)
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = EnergyForecastDashboard()
    
    dashboard = st.session_state.dashboard
    
    # Sidebar controls
    st.sidebar.markdown("### 🎛️ Controls")
    
    # Data generation options
    with st.sidebar.expander("📊 Data Settings", expanded=True):
        n_samples = st.slider("Number of samples", 1000, 20000, 10000, 1000)
        generate_data = st.button("🔄 Generate Sample Data")
    
    # Model training options
    with st.sidebar.expander("🤖 Model Settings", expanded=True):
        selected_models = st.multiselect(
            "Select models to train",
            ['Linear Regression', 'Ridge Regression', 'Random Forest', 
             'Gradient Boosting', 'XGBoost'],
            default=['Random Forest', 'XGBoost', 'Linear Regression']
        )
        train_models = st.button("🚀 Train Selected Models")
    
    # Main content
    st.markdown("<h1 class='main-header'>🏠 Household Energy Consumption Forecasting</h1>", unsafe_allow_html=True)
    
    # Generate data section
    if generate_data:
        with st.spinner("Generating sample data..."):
            dashboard.df = dashboard.generate_sample_data(n_samples)
            st.success(f"✅ Generated {len(dashboard.df)} samples of household energy data!")
            
            # Show data preview
            with st.expander("📋 View Sample Data", expanded=False):
                st.dataframe(dashboard.df.head(), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(dashboard.df))
                with col2:
                    st.metric("Features", len(dashboard.df.columns))
                with col3:
                    st.metric("Date Range", 
                             f"{dashboard.df['timestamp'].min().date()} to {dashboard.df['timestamp'].max().date()}")
    
    # Train models section
    if train_models and dashboard.df is not None:
        with st.spinner("Training models..."):
            # Engineer features using the fixed function
            X, y, features, df_processed = dashboard.engineer_features_fixed(dashboard.df)
            dashboard.features = features
            
            # Train models
            results, X_test, y_test = dashboard.train_models(X, y, features)
            dashboard.X_test = X_test
            dashboard.y_test = y_test
            
            st.success("✅ Model training completed!")
            
            # Show best model
            if results:
                best_model = max(results.items(), key=lambda x: x[1]['test_r2'])
                st.info(f"🏆 **Best Model**: {best_model[0]} with R² = {best_model[1]['test_r2']:.3f}")
    
    # Display dashboard
    if dashboard.results:
        dashboard.display_metrics_dashboard()
        
        # Recommendations section
        st.markdown("---")
        st.markdown("<h2 class='sub-header'>💡 Recommendations for R² Improvement</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔧 Technical Improvements
            1. **Feature Engineering**
               - Add external weather data
               - Include holiday calendar
               - Create more lag features
            2. **Hyperparameter Tuning**
               - Use Bayesian Optimization
               - Try different model architectures
               - Ensemble multiple models
            3. **Data Quality**
               - Handle outliers more effectively
               - Add more historical data
               - Include smart meter data
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Expected R² Gains
            - **Feature Engineering**: +0.10 to +0.25
            - **More Data**: +0.05 to +0.15
            - **Advanced Models**: +0.10 to +0.20
            - **Ensemble Methods**: +0.05 to +0.10
            - **Total Potential**: **+0.30 to +0.70**
            
            ### 🎯 Quick Wins
            1. Add temperature forecast data
            2. Include occupancy patterns
            3. Use holiday effects
            4. Add appliance-specific features
            """)
    
    elif dashboard.df is not None:
        st.info("👈 Click 'Train Selected Models' in the sidebar to start model training!")
    
    else:
        st.info("👈 Click 'Generate Sample Data' in the sidebar to get started!")

if __name__ == "__main__":
    main()
