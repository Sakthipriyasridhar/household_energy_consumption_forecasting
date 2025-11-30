import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="AI Consumption Forecasting",
    page_icon="🔮",
    layout="wide"
)

# Navigation
col_nav, col_title = st.columns([1, 5])
with col_nav:
    if st.button("← Back to Main"):
        st.switch_page("main.py")
with col_title:
    st.title("🔮 AI Consumption Forecasting")
    st.markdown("Predict future electricity bills using machine learning")

# UTILITY FUNCTIONS

def calculate_tneb_bill(units):
    """Calculate TNEB bill based on slab rates"""
    try:
        u = float(units)
    except:
        return 0.0
    if u <= 100:
        return 0.0
    remaining = u - 100.0
    bill = 0.0
    # Block 1: 101-300 units
    block = min(remaining, 200.0)
    bill += block * 4.50
    remaining -= block
    # Block 2: 301-500 units
    if remaining > 0:
        block = min(remaining, 200.0)
        bill += block * 6.00
        remaining -= block
    # Block 3: > 500 units
    if remaining > 0:
        bill += remaining * 8.00
    return round(bill, 2)


def create_rf_model():
    """Create OPTIMIZED Random Forest model for better R²"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(
            n_estimators=400,  # Increased from 200
            max_depth=20,  # Increased from 15
            min_samples_split=3,  # More flexible (was 5)
            min_samples_leaf=1,  # More flexible (was 2)
            max_features='sqrt',  # Better generalization
            bootstrap=True,  # Better performance
            random_state=42,
            n_jobs=-1
        ))
    ])


def add_comprehensive_features(monthly_data):
    """Enhanced feature engineering for better R² score"""
    df = monthly_data.copy()

    # Basic features (keep existing)
    df['Time_Index'] = range(len(df))
    df['Year_squared'] = df['Year'] ** 2

    # ENHANCED: More lag features
    for lag in [1, 2, 3, 4, 5, 6, 12]:  # Added more lags including yearly
        df[f'Units_lag_{lag}'] = df['Units'].shift(lag)

    # ENHANCED: Multiple rolling windows
    for window in [3, 6, 12]:
        df[f'Units_rolling_mean_{window}'] = df['Units'].rolling(window).mean()
        df[f'Units_rolling_std_{window}'] = df['Units'].rolling(window).std()

    # NEW: Seasonal features
    df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12)

    # NEW: Year-over-year growth
    df['yearly_growth'] = df['Units'].pct_change(12)

    # Handle infinite values from percentage change
    df = df.replace([np.inf, -np.inf], np.nan)

    return df.dropna()


def calculate_model_metrics(y_true, y_pred):
    """Calculate comprehensive model evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100

    return {
        'MAE': round(mae, 2),
        'MSE': round(mse, 2),
        'RMSE': round(rmse, 2),
        'R² Score': round(r2, 4),
        'MAPE (%)': round(mape, 2)
    }

# SIDEBAR - USER INPUTS

st.sidebar.header(" Forecast Configuration")

# Data Upload - Support multiple formats
uploaded_file = st.sidebar.file_uploader(
    "Upload Historical Data",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, Excel (.xlsx, .xls)"
)

if uploaded_file:
    # Read file based on extension
    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            st.stop()

        st.sidebar.success(f" File loaded successfully: {uploaded_file.name}")

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()

    # Column selection
    st.sidebar.subheader("Data Columns")
    cols = df.columns.tolist()

    # Auto-detect date and units columns
    date_candidates = [col for col in cols if any(term in col.lower() for term in ['date', 'time', 'day'])]
    units_candidates = [col for col in cols if
                        any(term in col.lower() for term in ['unit', 'consumption', 'kwh', 'energy', 'power'])]

    date_col = st.sidebar.selectbox(
        "Date Column",
        cols,
        index=cols.index(date_candidates[0]) if date_candidates else 0
    )

    units_col = st.sidebar.selectbox(
        "Units Column",
        cols,
        index=cols.index(units_candidates[0]) if units_candidates else 1
    )

    # Forecast parameters
    st.sidebar.subheader("Forecast Parameters")
    target_year = st.sidebar.number_input(
        "Target Year for Prediction",
        min_value=2024,
        max_value=2030,
        value=2025
    )

    # Model validation split
    test_size = st.sidebar.slider(
        "Validation Split (%)",
        min_value=10,
        max_value=40,
        value=20,
        help="Percentage of data to use for model validation"
    )

else:
    st.info(" Please upload your household energy consumption data to begin")
    st.stop()


# DATA PROCESSING

# Process main data
df_processed = df[[date_col, units_col]].copy()
df_processed = df_processed.rename(columns={date_col: 'Date', units_col: 'Units'})
df_processed['Date'] = pd.to_datetime(df_processed['Date'])
df_processed = df_processed.set_index('Date')
df_processed['Units'] = pd.to_numeric(df_processed['Units'], errors='coerce')
df_processed = df_processed.dropna()

# Create monthly data
monthly_data = df_processed['Units'].resample('M').sum().reset_index()
monthly_data['Year'] = monthly_data['Date'].dt.year
monthly_data['Month'] = monthly_data['Date'].dt.month
monthly_data['MonthName'] = monthly_data['Date'].dt.strftime('%b')

# MAIN INTERFACE - TABS

tab1, tab2, tab3, tab4 = st.tabs([" Data Overview", " Model Metrics", " Consumption Forecast", " Bill Prediction"])

with tab1:
    st.header("Data Overview & Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df_processed.head(10), use_container_width=True)

        # Basic statistics
        st.metric("Total Records", len(df_processed))
        st.metric("Date Range",
                  f"{df_processed.index.min().strftime('%Y-%m-%d')} to {df_processed.index.max().strftime('%Y-%m-%d')}")
        st.metric("Average Monthly Consumption", f"{monthly_data['Units'].mean():.1f} units")

        # Data quality check
        st.subheader("Data Quality")
        missing_values = df_processed['Units'].isnull().sum()
        zero_consumption = (df_processed['Units'] == 0).sum()

        col1a, col2a = st.columns(2)
        with col1a:
            st.metric("Missing Values", missing_values)
        with col2a:
            st.metric("Zero Consumption Days", zero_consumption)

    with col2:
        st.subheader("Consumption Trends")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Monthly trend
        monthly_data.groupby('Year')['Units'].sum().plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title("Annual Consumption Trend")
        ax1.set_ylabel("Total Units")

        # Seasonal pattern
        monthly_data.groupby('Month')['Units'].mean().plot(kind='line', ax=ax2, marker='o', color='coral')
        ax2.set_title("Average Monthly Consumption Pattern")
        ax2.set_ylabel("Average Units")
        ax2.set_xlabel("Month")

        plt.tight_layout()
        st.pyplot(fig)

with tab2:
    st.header(" Random Forest Model Performance")

    with st.spinner("Training and evaluating Enhanced Random Forest model..."):
        # ENHANCED: Use comprehensive feature engineering
        forecast_df = add_comprehensive_features(monthly_data)

        # Get all feature columns automatically
        features = [col for col in forecast_df.columns if col not in
                    ['Date', 'Units', 'MonthName', 'Year', 'Month']]

        if len(forecast_df) < 3:
            st.error("Insufficient data for model training. Need at least 3 months of historical data.")
        else:
            # Prepare training data
            X = forecast_df[features]
            y = forecast_df['Units']

            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42, shuffle=False
            )

            # Train model
            model = create_rf_model()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate metrics
            train_metrics = calculate_model_metrics(y_train, y_pred_train)
            test_metrics = calculate_model_metrics(y_test, y_pred_test)

            # Display metrics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(" Training Set Metrics")
                for metric, value in train_metrics.items():
                    if metric == 'R² Score':
                        st.metric(f"{metric}", f"{value:.4f}")
                    else:
                        st.metric(f"{metric}", f"{value}")

            with col2:
                st.subheader(" Test Set Metrics")
                for metric, value in test_metrics.items():
                    if metric == 'R² Score':
                        color = "normal"
                        if value > 0.8:
                            color = "green"
                        elif value > 0.6:
                            color = "orange"
                        else:
                            color = "red"
                        st.metric(f"{metric}", f"{value:.4f}")
                    else:
                        st.metric(f"{metric}", f"{value}")

            # Actual vs Predicted Plot
            st.subheader(" Actual vs Predicted Values")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Training set
            ax1.scatter(y_train, y_pred_train, alpha=0.6, color='blue')
            ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title(f'Training Set (R² = {train_metrics["R² Score"]:.4f})')

            # Test set
            ax2.scatter(y_test, y_pred_test, alpha=0.6, color='green')
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax2.set_xlabel('Actual Values')
            ax2.set_ylabel('Predicted Values')
            ax2.set_title(f'Test Set (R² = {test_metrics["R² Score"]:.4f})')

            plt.tight_layout()
            st.pyplot(fig)

            # Store the trained model for forecasting
            st.session_state.trained_model = model
            st.session_state.forecast_df = forecast_df
            st.session_state.features = features
            st.session_state.test_metrics = test_metrics

with tab3:
    st.header(" Future Consumption Forecast")

    if 'trained_model' not in st.session_state:
        st.warning("Please train the model in the 'Model Metrics' tab first.")
    else:
        with st.spinner("Generating future forecasts..."):
            model = st.session_state.trained_model
            forecast_df = st.session_state.forecast_df
            features = st.session_state.features

            # Generate future predictions
            future_predictions = []

            for year in [target_year]:
                for month in range(1, 13):
                    # Create future feature set
                    time_index = len(forecast_df) + len(future_predictions)
                    year_squared = year ** 2

                    # ENHANCED: Create comprehensive feature set
                    future_feature_dict = {
                        'Time_Index': time_index,
                        'Year': year,
                        'Month': month,
                        'Year_squared': year_squared,
                        'sin_month': np.sin(2 * np.pi * month / 12),
                        'cos_month': np.cos(2 * np.pi * month / 12),
                    }

                    # Add lag features (simplified - use available history)
                    for lag in [1, 2, 3, 4, 5, 6, 12]:
                        if lag <= len(future_predictions):
                            future_feature_dict[f'Units_lag_{lag}'] = future_predictions[-lag]['Predicted_Units']
                        else:
                            # Use historical data for initial predictions
                            historical_lag = forecast_df['Units'].iloc[-lag] if len(forecast_df) >= lag else \
                            forecast_df['Units'].mean()
                            future_feature_dict[f'Units_lag_{lag}'] = historical_lag

                    # Add rolling features (simplified)
                    for window in [3, 6, 12]:
                        if len(future_predictions) >= window:
                            recent_values = [fp['Predicted_Units'] for fp in future_predictions[-window:]]
                            future_feature_dict[f'Units_rolling_mean_{window}'] = np.mean(recent_values)
                            future_feature_dict[f'Units_rolling_std_{window}'] = np.std(recent_values)
                        else:
                            # Use historical rolling stats
                            if f'Units_rolling_mean_{window}' in forecast_df.columns:
                                future_feature_dict[f'Units_rolling_mean_{window}'] = \
                                forecast_df[f'Units_rolling_mean_{window}'].iloc[-1]
                                future_feature_dict[f'Units_rolling_std_{window}'] = \
                                forecast_df[f'Units_rolling_std_{window}'].iloc[-1]
                            else:
                                future_feature_dict[f'Units_rolling_mean_{window}'] = forecast_df['Units'].mean()
                                future_feature_dict[f'Units_rolling_std_{window}'] = forecast_df['Units'].std()

                    # Handle yearly growth (simplified)
                    if len(future_predictions) >= 12:
                        future_feature_dict['yearly_growth'] = (future_predictions[-1]['Predicted_Units'] -
                                                                future_predictions[-12]['Predicted_Units']) / \
                                                               future_predictions[-12]['Predicted_Units']
                    else:
                        future_feature_dict['yearly_growth'] = forecast_df['yearly_growth'].iloc[
                            -1] if 'yearly_growth' in forecast_df.columns else 0

                    future_features = pd.DataFrame([future_feature_dict])

                    # Ensure all training features are present
                    missing_cols = set(features) - set(future_features.columns)
                    for col in missing_cols:
                        future_features[col] = forecast_df[col].mean() if col in forecast_df.columns else 0

                    future_features = future_features[features]

                    # Predict
                    predicted_units = max(0, model.predict(future_features)[0])

                    future_predictions.append({
                        'Year': year,
                        'Month': month,
                        'MonthName': datetime(year, month, 1).strftime('%b'),
                        'Predicted_Units': round(predicted_units, 1),
                        'Bill_Amount': calculate_tneb_bill(predicted_units)
                    })

            # Create results dataframe
            results_df = pd.DataFrame(future_predictions)

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f" {target_year} Monthly Forecast")
                display_df = results_df[['MonthName', 'Predicted_Units', 'Bill_Amount']].copy()
                display_df['Bill_Amount'] = display_df['Bill_Amount'].apply(lambda x: f'₹{x:,.0f}')
                st.dataframe(display_df, use_container_width=True)

            with col2:
                st.subheader(" Forecast Summary")
                total_units = results_df['Predicted_Units'].sum()
                total_bill = results_df['Bill_Amount'].sum()
                avg_monthly = total_bill / 12

                st.metric("Total Annual Units", f"{total_units:,.0f} kWh")
                st.metric("Total Annual Bill", f"₹{total_bill:,.0f}")
                st.metric("Average Monthly Bill", f"₹{avg_monthly:,.0f}")

            # Visualization
            st.subheader("Forecast Visualization")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Units forecast
            ax1.bar(results_df['MonthName'], results_df['Predicted_Units'], color='lightblue', alpha=0.7)
            ax1.set_title(f"Monthly Consumption Forecast - {target_year}")
            ax1.set_ylabel("Units (kWh)")
            ax1.tick_params(axis='x', rotation=45)

            # Bill forecast
            ax2.bar(results_df['MonthName'], results_df['Bill_Amount'], color='lightcoral', alpha=0.7)
            ax2.set_title(f"Monthly Bill Forecast - {target_year}")
            ax2.set_ylabel("Bill Amount (₹)")
            ax2.tick_params(axis='x', rotation=45)

            # Add value labels
            for i, v in enumerate(results_df['Bill_Amount']):
                ax2.text(i, v, f'₹{v:.0f}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)

            # Store results for next tab
            st.session_state.results_df = results_df

with tab4:
    st.header(" Detailed Bill Analysis")

    if 'results_df' not in st.session_state:
        st.warning("Please generate forecasts in the 'Consumption Forecast' tab first.")
    else:
        results_df = st.session_state.results_df

        # Bill analysis
        st.subheader("TNEB Bill Breakdown")

        # Calculate slab analysis
        slab_analysis = []
        for _, row in results_df.iterrows():
            units = row['Predicted_Units']
            bill_details = {
                'Month': row['MonthName'],
                'Units': units,
                'Total_Bill': f"₹{row['Bill_Amount']:,.0f}"
            }

            slab_analysis.append(bill_details)

        slab_df = pd.DataFrame(slab_analysis)
        st.dataframe(slab_df, use_container_width=True)

        # Annual summary
        st.subheader(" Annual Financial Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_units = results_df['Predicted_Units'].sum()
            st.metric("Total Consumption", f"{total_units:,.0f} kWh")

        with col2:
            total_bill = results_df['Bill_Amount'].sum()
            st.metric("Total Annual Cost", f"₹{total_bill:,.0f}")

        with col3:
            avg_monthly_bill = total_bill / 12
            st.metric("Monthly Average", f"₹{avg_monthly_bill:,.0f}")

        with col4:
            peak_month = results_df.loc[results_df['Predicted_Units'].idxmax()]
            st.metric("Peak Consumption Month", f"{peak_month['MonthName']}")

        # TNEB Slab Information
        st.subheader("ℹ TNEB Slab Rates")

        slab_info = pd.DataFrame({
            "Slab Range": ["0-100 units", "101-300 units", "301-500 units", "501+ units"],
            "Rate (₹/unit)": ["Free", "4.50", "6.00", "8.00"],
            "Description": [
                "Completely free for first 100 units",
                "₹4.50 per unit for 101-300 range",
                "₹6.00 per unit for 301-500 range",
                "₹8.00 per unit beyond 500 units"
            ]
        })

        st.dataframe(slab_info, use_container_width=True)

# EXPORT RESULTS

st.markdown("---")
st.header(" Export Results")

if 'results_df' in locals():
    col1, col2 = st.columns(2)

    with col1:
        # CSV Export
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name=f"energy_forecast_{target_year}.csv",
            mime="text/csv"
        )

    with col2:
        # Summary report
        total_units = results_df['Predicted_Units'].sum()
        total_bill = results_df['Bill_Amount'].sum()
        avg_monthly_bill = total_bill / 12
        test_metrics = st.session_state.get('test_metrics', {})

        st.download_button(
            label="Download Summary Report",
            data=f"""
            HOUSEHOLD ENERGY CONSUMPTION FORECAST - {target_year}

            Total Annual Units: {total_units:,.0f} kWh
            Total Annual Bill: ₹{total_bill:,.0f}
            Average Monthly Bill: ₹{avg_monthly_bill:,.0f}

            Forecast Method: Enhanced Random Forest Regression
            Model R² Score: {test_metrics.get('R² Score', 'N/A')}
            Model RMSE: {test_metrics.get('RMSE', 'N/A')}
            Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            """,
            file_name=f"energy_forecast_summary_{target_year}.txt",
            mime="text/plain"
        )

# --------------------------
# FOOTER
# --------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🏠 Enhanced Household Energy Consumption Forecasting </p>
    </div>
    """,
    unsafe_allow_html=True
)