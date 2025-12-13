import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ForecastingModelTrainer:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.forecasts = {}
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and store results"""
        
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
        }
        
        for name, model in models_to_train.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_test, y_pred)
                
                # Store results
                self.models[name] = model
                self.metrics[name] = metrics
                self.forecasts[name] = y_pred
                
                st.success(f"✅ {name} trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Error training {name}: {str(e)}")
                continue
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE safely (avoid division by zero)
        y_true_nonzero = np.where(y_true == 0, 1e-10, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_nonzero)) * 100
        
        return {
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R²': round(r2, 4),
            'MAPE (%)': round(mape, 2)
        }
    
    def generate_forecast(self, model_name, future_X):
        """Generate forecast using trained model"""
        if model_name in self.models:
            return self.models[model_name].predict(future_X)
        return None

def main():
    st.title("📊 Multi-Algorithm Forecasting Dashboard")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        selected_models = st.multiselect(
            "Select Forecasting Models:",
            ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
             'Random Forest', 'Gradient Boosting', 'SVR', 'K-Nearest Neighbors'],
            default=['Linear Regression', 'Random Forest', 'Gradient Boosting']
        )
        
        # Forecast horizon
        forecast_days = st.slider("Forecast Period (days):", 1, 365, 30)
        
        # Lag features
        n_lags = st.slider("Number of Lag Features:", 1, 50, 10)
        
        # Train/test split
        test_size = st.slider("Test Size (%):", 10, 50, 20) / 100
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Data Visualization")
        
        # Load your data here
        # data = pd.read_csv('your_data.csv')
        # For now, create sample data
        np.random.seed(42)
        n_samples = 1000
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        values = np.cumsum(np.random.randn(n_samples)) + 100
        data = pd.DataFrame({'timestamp': dates, 'value': values})
        
        # Plot original data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['timestamp'], y=data['value'],
                                mode='lines', name='Original Data',
                                line=dict(color='blue', width=1)))
        fig.update_layout(title='Original Time Series Data',
                         xaxis_title='Date',
                         yaxis_title='Value',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("📊 Quick Stats")
        st.metric("Total Samples", len(data))
        st.metric("Mean Value", f"{data['value'].mean():.2f}")
        st.metric("Std Deviation", f"{data['value'].std():.2f}")
    
    # Create lag features
    st.subheader("🔧 Feature Engineering")
    with st.expander("Create Lag Features"):
        for lag in range(1, n_lags + 1):
            data[f'lag_{lag}'] = data['value'].shift(lag)
        
        # Drop NaN values created by shifting
        data_clean = data.dropna()
        st.write(f"Data shape after creating {n_lags} lag features: {data_clean.shape}")
        st.dataframe(data_clean.head(), use_container_width=True)
    
    # Prepare data for training
    X = data_clean.drop(['timestamp', 'value'], axis=1)
    y = data_clean['value']
    
    # Split data
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Initialize trainer
    trainer = ForecastingModelTrainer()
    
    # Train models button
    if st.button("🚀 Train All Selected Models", type="primary"):
        with st.spinner("Training models..."):
            # Train selected models
            models_to_train = {}
            if 'Linear Regression' in selected_models:
                models_to_train['Linear Regression'] = LinearRegression()
            if 'Ridge Regression' in selected_models:
                models_to_train['Ridge Regression'] = Ridge(alpha=1.0)
            if 'Lasso Regression' in selected_models:
                models_to_train['Lasso Regression'] = Lasso(alpha=0.1)
            if 'Random Forest' in selected_models:
                models_to_train['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
            if 'Gradient Boosting' in selected_models:
                models_to_train['Gradient Boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
            if 'SVR' in selected_models:
                models_to_train['SVR'] = SVR(kernel='rbf')
            if 'K-Nearest Neighbors' in selected_models:
                models_to_train['K-Nearest Neighbors'] = KNeighborsRegressor(n_neighbors=5)
            
            # Train models
            for name, model in models_to_train.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = trainer.calculate_metrics(y_test, y_pred)
                    
                    trainer.models[name] = model
                    trainer.metrics[name] = metrics
                    trainer.forecasts[name] = y_pred
                    
                except Exception as e:
                    st.error(f"Error training {name}: {str(e)}")
            
            st.success("✅ All models trained successfully!")
    
    # Display results in tabs
    if trainer.models:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Model Comparison", 
            "📈 Forecast Visualization", 
            "📋 Detailed Metrics",
            "📥 Download Results"
        ])
        
        with tab1:
            st.subheader("Model Performance Comparison")
            
            # Create comparison table
            comparison_df = pd.DataFrame(trainer.metrics).T
            st.dataframe(comparison_df.style
                        .background_gradient(subset=['R²'], cmap='Greens')
                        .background_gradient(subset=['RMSE', 'MAE'], cmap='Reds_r')
                        .format("{:.4f}"), 
                        use_container_width=True)
            
            # Visual comparison
            fig = make_subplots(rows=2, cols=2,
                               subplot_titles=('R² Score', 'RMSE', 'MAE', 'MAPE (%)'))
            
            models = list(trainer.metrics.keys())
            r2_scores = [trainer.metrics[m]['R²'] for m in models]
            rmse_scores = [trainer.metrics[m]['RMSE'] for m in models]
            mae_scores = [trainer.metrics[m]['MAE'] for m in models]
            mape_scores = [trainer.metrics[m]['MAPE (%)'] for m in models]
            
            fig.add_trace(go.Bar(x=models, y=r2_scores, name='R²'), row=1, col=1)
            fig.add_trace(go.Bar(x=models, y=rmse_scores, name='RMSE'), row=1, col=2)
            fig.add_trace(go.Bar(x=models, y=mae_scores, name='MAE'), row=2, col=1)
            fig.add_trace(go.Bar(x=models, y=mape_scores, name='MAPE'), row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Forecast Visualization")
            
            # Model selector for visualization
            selected_model = st.selectbox(
                "Select Model to Visualize:",
                list(trainer.models.keys())
            )
            
            if selected_model:
                y_pred = trainer.forecasts[selected_model]
                
                # Create forecast visualization
                fig = go.Figure()
                
                # Actual values
                fig.add_trace(go.Scatter(
                    x=list(range(len(y_test))),
                    y=y_test.values,
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue', width=2)
                ))
                
                # Predicted values
                fig.add_trace(go.Scatter(
                    x=list(range(len(y_pred))),
                    y=y_pred,
                    mode='lines',
                    name=f'{selected_model} Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f'{selected_model} Forecast vs Actual',
                    xaxis_title='Time Index',
                    yaxis_title='Value',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Residual plot
                residuals = y_test.values - y_pred
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=y_pred,
                    y=residuals,
                    mode='markers',
                    marker=dict(size=8, opacity=0.6)
                ))
                fig2.add_hline(y=0, line_dash="dash", line_color="red")
                fig2.update_layout(
                    title='Residual Plot',
                    xaxis_title='Predicted Values',
                    yaxis_title='Residuals',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.subheader("Detailed Model Metrics")
            
            for model_name in trainer.models.keys():
                with st.expander(f"📊 {model_name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Model Parameters:**")
                        if hasattr(trainer.models[model_name], 'get_params'):
                            params = trainer.models[model_name].get_params()
                            for key, value in params.items():
                                st.text(f"{key}: {value}")
                    
                    with col2:
                        st.markdown("**Performance Metrics:**")
                        metrics = trainer.metrics[model_name]
                        for metric_name, value in metrics.items():
                            st.metric(metric_name, value)
        
        with tab4:
            st.subheader("Download Results")
            
            # Prepare download data
            results_data = {}
            for model_name in trainer.models.keys():
                results_data[f'{model_name}_forecast'] = trainer.forecasts[model_name]
            
            results_df = pd.DataFrame(results_data)
            results_df['Actual'] = y_test.values
            
            # Convert to CSV
            csv = results_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Forecast Results (CSV)",
                data=csv,
                file_name="forecast_results.csv",
                mime="text/csv"
            )
            
            # Download metrics
            metrics_df = pd.DataFrame(trainer.metrics).T
            metrics_csv = metrics_df.to_csv()
            
            st.download_button(
                label="📥 Download Metrics (CSV)",
                data=metrics_csv,
                file_name="model_metrics.csv",
                mime="text/csv"
            )
    
    # Model recommendations
    st.sidebar.header("🎯 Recommendations")
    if trainer.metrics:
        best_model = max(trainer.metrics.items(), key=lambda x: x[1]['R²'])[0]
        st.sidebar.success(f"**Recommended Model:** {best_model}")
        st.sidebar.info(f"**Best R²:** {trainer.metrics[best_model]['R²']:.4f}")

if __name__ == "__main__":
    main()
