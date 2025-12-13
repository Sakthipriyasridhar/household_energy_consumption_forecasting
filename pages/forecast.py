import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-excellent {
        background-color: #E8F5E9;
        color: #2E7D32;
        border: 1px solid #A5D6A7;
    }
    .badge-good {
        background-color: #E3F2FD;
        color: #1565C0;
        border: 1px solid #90CAF9;
    }
    .badge-warning {
        background-color: #FFF3E0;
        color: #EF6C00;
        border: 1px solid #FFCC80;
    }
    .section-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 1.5rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
    }
    .algorithm-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
        cursor: pointer;
    }
    .algorithm-card:hover {
        border-color: #1E88E5;
        box-shadow: 0 2px 8px rgba(30, 136, 229, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_algorithms' not in st.session_state:
    st.session_state.selected_algorithms = []
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">📈 Mouseholder Forecasting Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Forecast your time series data with machine learning algorithms")

with col2:
    st.markdown("")
    if st.button("🔄 Run Forecast", use_container_width=True):
        # Simulate forecast generation
        st.session_state.forecast_data = generate_sample_forecast()
        st.session_state.sample_data = generate_sample_data()
        st.success("Forecast completed successfully!")

# Main content in tabs
tab1, tab2, tab3 = st.tabs(["📊 Data & Algorithms", "📈 Forecast Results", "📋 Performance"])

with tab1:
    # Data Upload Section
    st.markdown('<div class="sub-header">📁 Upload Your Data</div>', unsafe_allow_html=True)
    
    upload_col1, upload_col2 = st.columns(2)
    
    with upload_col1:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", 
                                         help="Upload your time series data in CSV format")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.sample_data = df
                st.success(f"✅ Data loaded successfully! ({len(df)} rows, {len(df.columns)} columns)")
                
                # Show data preview
                with st.expander("📋 View Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error loading file: {e}")
        else:
            # Show sample data option
            if st.button("📂 Load Sample Dataset", use_container_width=True):
                st.session_state.sample_data = generate_sample_data()
                st.success("Sample data loaded successfully!")
    
    with upload_col2:
        if st.session_state.sample_data is not None:
            df = st.session_state.sample_data
            st.markdown("**Data Statistics:**")
            col1_stat, col2_stat, col3_stat = st.columns(3)
            with col1_stat:
                st.metric("Rows", len(df))
            with col2_stat:
                st.metric("Columns", len(df.columns))
            with col3_stat:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Column selector for target
            if len(df.columns) > 0:
                target_col = st.selectbox("Select Target Column", df.columns)
                st.session_state.target_column = target_col

    # Algorithm Selection Section
    st.markdown('<div class="sub-header">🤖 Select Algorithms</div>', unsafe_allow_html=True)
    
    st.markdown("### Question Tree")
    
    # Algorithm categories
    categories = {
        "Linear Models": ["Linear Regression"],
        "Instance-Based": ["K-Nearest Neighbors"],
        "Kernel Methods": ["Support Vector Regression"],
        "Ensemble": ["AdaBoost"],
        "Time Series": ["ARIMA", "Exponential Smoothing"]
    }
    
    # Create algorithm selection
    selected_algorithms = []
    
    cols = st.columns(len(categories))
    for idx, (category, algorithms) in enumerate(categories.items()):
        with cols[idx]:
            st.markdown(f"**{category}**")
            for algo in algorithms:
                if st.checkbox(f"{algo}", key=f"algo_{algo}"):
                    selected_algorithms.append(algo)
    
    # Quick select buttons
    st.markdown("### Quick Select")
    col_quick1, col_quick2, col_quick3 = st.columns([1, 1, 2])
    
    with col_quick1:
        if st.button("✓ Select All", use_container_width=True):
            for category in categories.values():
                for algo in category:
                    st.session_state[f"algo_{algo}"] = True
    
    with col_quick2:
        if st.button("✗ Clear All", use_container_width=True):
            for category in categories.values():
                for algo in category:
                    st.session_state[f"algo_{algo}"] = False
    
    with col_quick3:
        st.info(f"Selected: {len(selected_algorithms)} algorithms")

    # Algorithm Info
    if selected_algorithms:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown("### 📚 Selected Algorithm Info")
        for algo in selected_algorithms:
            info = get_algorithm_info(algo)
            with st.expander(f"ℹ️ {algo}", expanded=False):
                st.markdown(f"**Category:** {info['category']}")
                st.markdown(f"**Type:** {info['type']}")
                st.markdown(f"**Description:** {info['description']}")
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # Forecast Results Section
    st.markdown('<div class="sub-header">📈 Forecast Visualization</div>', unsafe_allow_html=True)
    
    if st.session_state.forecast_data is not None:
        forecast_df = st.session_state.forecast_data
        
        # Create interactive plot
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Forecasted Values', 'Prediction Intervals'),
                           vertical_spacing=0.15,
                           row_heights=[0.7, 0.3])
        
        # Add actual vs predicted
        fig.add_trace(
            go.Scatter(x=forecast_df['date'], y=forecast_df['actual'], 
                      name='Actual', line=dict(color='#1E88E5', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=forecast_df['date'], y=forecast_df['predicted'], 
                      name='Predicted', line=dict(color='#FF6B6B', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(x=forecast_df['date'], y=forecast_df['upper_bound'],
                      fill=None, mode='lines', line_color='rgba(255, 107, 107, 0.2)',
                      showlegend=False, name='Upper Bound'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=forecast_df['date'], y=forecast_df['lower_bound'],
                      fill='tonexty', mode='lines', fillcolor='rgba(255, 107, 107, 0.2)',
                      line_color='rgba(255, 107, 107, 0.2)', showlegend=False, name='Lower Bound'),
            row=1, col=1
        )
        
        # Add error bars
        fig.add_trace(
            go.Bar(x=forecast_df['date'], y=forecast_df['error'],
                  name='Error', marker_color='#FFA726'),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(height=700, showlegend=True, 
                         template='plotly_white',
                         hovermode='x unified')
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Error", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast data table
        st.markdown("### 📋 Forecast Data")
        st.dataframe(forecast_df.style.format({
            'actual': '{:.2f}',
            'predicted': '{:.2f}',
            'error': '{:.2f}',
            'upper_bound': '{:.2f}',
            'lower_bound': '{:.2f}'
        }), use_container_width=True)
        
        # Download button
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast Data (CSV)",
            data=csv,
            file_name="forecast_results.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        st.warning("⚠️ No forecast data available. Please run a forecast first.")
        st.info("Go to the 'Data & Algorithms' tab to upload data and select algorithms, then click 'Run Forecast'.")

with tab3:
    # Performance Metrics Section
    st.markdown('<div class="sub-header">📊 Performance Metrics</div>', unsafe_allow_html=True)
    
    if st.session_state.forecast_data is not None:
        # Calculate metrics
        metrics = calculate_metrics(st.session_state.forecast_data)
        
        # Display metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.metric("R² Score", f"{metrics['r2']:.3f}", 
                     delta_color="normal")
            st.markdown('<small>Coefficient of Determination</small>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            color_class = "badge-excellent" if metrics['rmse'] < 10 else "badge-warning"
            st.markdown(f'''
            <div class="section-box">
                <div class="metric-badge {color_class}">RMSE: {metrics['rmse']:.3f}</div>
                <small>Root Mean Square Error</small>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            color_class = "badge-excellent" if metrics['mae'] < 8 else "badge-warning"
            st.markdown(f'''
            <div class="section-box">
                <div class="metric-badge {color_class}">MAE: {metrics['mae']:.3f}</div>
                <small>Mean Absolute Error</small>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            color_class = "badge-excellent" if metrics['mape'] < 5 else "badge-warning"
            st.markdown(f'''
            <div class="section-box">
                <div class="metric-badge {color_class}">MAPE: {metrics['mape']:.1f}%</div>
                <small>Mean Absolute Percentage Error</small>
            </div>
            ''', unsafe_allow_html=True)
        
        # Algorithm Comparison Chart
        st.markdown("### 📊 Algorithm Performance Comparison")
        
        # Simulate different algorithm performances
        algorithms = ["Linear Regression", "K-Nearest Neighbors", "Support Vector Regression", "AdaBoost", "ARIMA"]
        performances = np.random.uniform(0.85, 1.0, len(algorithms))
        
        perf_df = pd.DataFrame({
            'Algorithm': algorithms,
            'R² Score': performances,
            'RMSE': np.random.uniform(0.5, 5.0, len(algorithms)),
            'MAE': np.random.uniform(0.3, 4.0, len(algorithms))
        })
        
        # Create bar chart
        fig_perf = px.bar(perf_df, x='Algorithm', y='R² Score',
                         color='R² Score', color_continuous_scale='Viridis',
                         title='Algorithm R² Scores Comparison',
                         text_auto='.3f')
        
        fig_perf.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Performance table
        st.markdown("### 📋 Detailed Metrics")
        st.dataframe(perf_df.style.format({
            'R² Score': '{:.3f}',
            'RMSE': '{:.3f}',
            'MAE': '{:.3f}'
        }).background_gradient(subset=['R² Score'], cmap='RdYlGn'), 
        use_container_width=True)
        
    else:
        st.warning("⚠️ No performance data available. Please run a forecast first.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Forecasting Dashboard • Powered by Streamlit • All algorithms are for demonstration purposes</p>
</div>
""", unsafe_allow_html=True)

# Helper functions
def generate_sample_data():
    """Generate sample time series data"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    trend = np.linspace(100, 200, len(dates))
    seasonal = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    noise = np.random.normal(0, 10, len(dates))
    values = trend + seasonal + noise
    
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'feature1': np.random.uniform(0, 100, len(dates)),
        'feature2': np.random.uniform(0, 50, len(dates)),
        'feature3': np.random.randint(0, 10, len(dates))
    })
    return df

def generate_sample_forecast():
    """Generate sample forecast data"""
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    np.random.seed(42)
    
    base_trend = np.linspace(200, 250, len(dates))
    seasonal = 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 90)
    noise = np.random.normal(0, 8, len(dates))
    
    actual = base_trend + seasonal + noise
    predicted = actual + np.random.normal(0, 5, len(dates))
    error = predicted - actual
    
    forecast_df = pd.DataFrame({
        'date': dates,
        'actual': actual,
        'predicted': predicted,
        'error': error,
        'upper_bound': predicted + 15,
        'lower_bound': predicted - 15
    })
    
    return forecast_df

def get_algorithm_info(algorithm_name):
    """Get information about an algorithm"""
    info = {
        'Linear Regression': {
            'category': 'Linear Models',
            'type': 'Supervised Regression',
            'description': 'Simple linear relationship between features and target'
        },
        'K-Nearest Neighbors': {
            'category': 'Instance-Based',
            'type': 'Supervised Regression',
            'description': 'Predicts based on k most similar training examples'
        },
        'Support Vector Regression': {
            'category': 'Kernel Methods',
            'type': 'Supervised Regression',
            'description': 'Uses support vectors to find optimal regression line'
        },
        'AdaBoost': {
            'category': 'Ensemble',
            'type': 'Supervised Regression',
            'description': 'Adaptive boosting that combines multiple weak learners'
        },
        'ARIMA': {
            'category': 'Time Series',
            'type': 'Unsupervised',
            'description': 'Autoregressive Integrated Moving Average for time series'
        },
        'Exponential Smoothing': {
            'category': 'Time Series',
            'type': 'Unsupervised',
            'description': 'Weighted average of past observations'
        }
    }
    
    return info.get(algorithm_name, {
        'category': 'Unknown',
        'type': 'Unknown',
        'description': 'No description available'
    })

def calculate_metrics(forecast_df):
    """Calculate performance metrics from forecast data"""
    actual = forecast_df['actual'].values
    predicted = forecast_df['predicted'].values
    
    # Calculate metrics
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Calculate R²
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
