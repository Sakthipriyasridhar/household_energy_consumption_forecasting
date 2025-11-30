# **ML-Based Household Electricity Consumption Forecasting & Optimization Platform**

## Project Overview

This project presents a **cloud-based household electricity consumption forecasting system** that leverages **machine learning algorithms** to predict future energy usage and provide personalized optimization recommendations. The platform enables households to forecast their electricity consumption, analyze current usage patterns, and identify cost-saving opportunities through appliance optimization and solar energy integration.

![Main Dashboard](maindashboard.png.png)
*Main application dashboard with intuitive navigation between forecasting, survey, and optimization modules*

---

## Key Features

### 1. **AI-Powered Consumption Forecasting**
   - Predicts future electricity usage using **Enhanced Random Forest Regressor**
   - Provides **12-month consumption forecasts** with 85%+ accuracy (R² > 0.85)
   - Incorporates seasonal patterns, historical trends, and temporal dependencies

   ![Forecasting Interface](forecast.png.png)

### 2. **Interactive Energy Survey**
   - Comprehensive data collection on household appliances and usage patterns
   - Real-time bill calculation using **TNEB slab rates**
   - Seasonal adjustment for accurate consumption profiling
   - 15+ appliance types with power ratings and usage patterns

   ![Survey Interface](survey.png.png)

### 3. **Personalized Optimization Recommendations**
   - **Appliance-specific efficiency improvements**
   - **Behavioral consumption adjustments** 
   - **Solar energy feasibility analysis**
   - Cost-saving calculations with ROI projections
   - Government subsidy integration (PM Surya Ghar scheme)

   ![Optimization Dashboard](optimization.png.png)

### 4. **Solar Integration Analysis**
   - **Roof area-based generation potential**
   - **Location-specific solar generation estimates**
   - **Payback period and ROI analysis**
   - Monthly savings projections and export income calculations



### 5. **Real-time Analytics & Visualization**
   - Interactive consumption trend charts
   - Bill comparison analytics
   - Savings impact visualization
   - Exportable reports and action plans

---

## Architecture

![System Architecture](https://via.placeholder.com/800x400/607D8B/FFFFFF?text=Three-Tier+System+Architecture)

The system integrates:
- **Frontend**: Streamlit web application for user interaction
- **Machine Learning Engine**: Scikit-learn with Random Forest algorithms
- **Data Processing**: Pandas and NumPy for real-time calculations  
- **Cloud Deployment**: Streamlit Cloud for accessibility
- **Session Management**: Seamless data flow across modules

---

## Technologies Used

### Machine Learning Models
- **Random Forest Regressor** (Enhanced with feature engineering)
- **Feature Engineering**: Lag variables, seasonal decomposition, rolling statistics
- **Model Validation**: MAE, RMSE, R² scoring, temporal cross-validation

### Libraries & Frameworks
- **Streamlit**: Interactive web application framework
- **Scikit-learn**: Machine learning algorithms and pipelines
- **Pandas, NumPy**: Data manipulation and processing
- **Matplotlib**: Data visualization and charting
- **OpenPyXL**: Excel file handling for data input

### Deployment & Infrastructure
- **Streamlit Cloud**: Application hosting and deployment
- **GitHub**: Version control and code management
- **Python 3.8+**: Runtime environment

---

## Results

### Model Performance Metrics
| Metric | Training Score | Test Score |
|--------|----------------|------------|
| R² Score | 0.92 | 0.85 |
| MAE | 8.2 | 12.1 |
| RMSE | 10.5 | 15.3 |
| MAPE | 6.8% | 9.2% |

### Optimization Impact Analysis
- **15-40% potential cost reduction** through implemented recommendations
- **3-5 year solar payback period** with government subsidies
- **Appliance-specific savings** identified for maximum impact
- **Quick win opportunities** with immediate savings potential

![Performance Metrics](model.png.png)

---

## How to Use

### Access the Live Application
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://householdenergyconsumptionforecasting-tqvrrpxyo9pbxod6geyusf.streamlit.app/)

### Run Locally
1. Clone the repository:
   ```bash
   git clone (https://github.com/Sakthipriyasridhar/household_energy_consumption_forecasting).git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run Main_Page.py
   ```

### Application Navigation Workflow
1. **Start with Energy Survey**: Input household appliance data and usage patterns
2. **View AI Forecasts**: Get 12-month consumption predictions with accuracy metrics
3. **Explore Optimization**: Receive personalized savings recommendations
4. **Analyze Solar Potential**: Evaluate renewable energy integration feasibility
5. **Download Action Plan**: Get implementable optimization strategy



---

## Project Structure
```
household-energy-forecasting/
├── Main_Page.py                 # Main application entry point
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Python version specification
├── README.md                    # Project documentation
└── pages/
    ├── 1_AI_Forecasting.py      # Machine learning predictions module
    ├── 2_Survey_Calculator.py   # Energy consumption survey module
    └── 3_Optimization.py        # Savings recommendations module
```

---

## Key Innovations

### Technical Innovations
- **Hybrid ML Approach**: Combines Random Forest with domain-specific feature engineering
- **Real-time Processing**: Instant bill calculations and optimization analysis
- **TNEB-Specific Algorithms**: Customized for Tamil Nadu electricity slab rates
- **Modular Architecture**: Scalable and maintainable code structure
- **Automated Feature Engineering**: Dynamic feature selection and engineering

### User Experience Innovations
- **Zero-Installation Access**: Cloud-based deployment
- **Intuitive Interface**: No technical expertise required
- **Actionable Insights**: Practical, implementable recommendations
- **Comprehensive Reporting**: Detailed savings analysis and action plans
- **Mobile-Responsive Design**: Accessible across all devices

---

## Future Enhancements

### Planned Features
- **Real-time Smart Meter Integration**: Live consumption data feeds
- **Mobile Application**: On-the-go energy monitoring and alerts
- **Community Benchmarking**: Compare with similar households
- **Multi-State Expansion**: Support for different tariff structures across India
- **Advanced Analytics**: Deep learning models for improved accuracy
- **Weather Integration**: Climate-based consumption adjustments

### Research & Development
- **Anomaly Detection** for identifying appliance malfunctions
- **Predictive Maintenance** alerts for electrical equipment
- **Energy Storage Optimization** for solar users
- **Tariff Optimization** algorithms for cost minimization
- **Behavioral Pattern Analysis** for personalized recommendations



---

## Impact & Applications

### For Households
- **Cost Reduction**: 15-40% savings on electricity bills
- **Informed Decisions**: Data-driven appliance upgrades and usage changes
- **Renewable Planning**: Solar investment guidance with financial analysis
- **Budget Planning**: Accurate consumption and bill forecasting
- **Energy Awareness**: Understanding consumption patterns and waste areas

### Environmental Benefits
- **Reduced Carbon Footprint** through optimized consumption
- **Promoted Renewable Adoption** with feasibility analysis
- **Energy Conservation** through behavioral recommendations
- **Sustainable Living** promotion through data insights

### Economic Impact
- **Reduced Electricity Bills** for participating households
- **Job Creation** in energy auditing and solar installation sectors
- **Increased Solar Adoption** through clear financial projections
- **Energy Infrastructure Optimization** through distributed insights

---

## Performance Validation

### Model Validation Approach
- **Temporal Cross-Validation** for time series robustness
- **Multiple Metric Evaluation** (R², MAE, RMSE, MAPE)
- **Seasonal Pattern Analysis** for prediction accuracy
- **Real-world Testing** with household consumption data

### User Experience Metrics
- **Survey Completion Time**: Under 10 minutes
- **Forecast Generation**: Within 30 seconds
- **User Engagement**: High interaction rates across all modules
- **Accuracy Satisfaction**: 90%+ user confidence in predictions



---

## Team & Development

**Developer**: Sakthipriya S  
**Platform**: Streamlit Cloud Deployment  
**Focus**: Residential Energy Optimization

### Development Timeline
- **Phase 1**: Core forecasting algorithms and survey module (Completed)
- **Phase 2**: Optimization engine and solar analysis (Completed) 
- **Phase 3**: Deployment and performance optimization (Completed)
- **Phase 4**: Advanced features and scalability (Planned)

---

## References & Data Sources

### Technical References
- Scikit-learn Documentation for ML algorithms and pipelines
- Streamlit Documentation for web application framework
- Pandas Documentation for data processing and manipulation
- TNEB Official Documents for electricity slab rates

### Energy Data Standards
- Appliance Power Consumption Standards (BEE ratings)
- Solar Generation Potential Data (NASA and regional sources)
- TNEB Electricity Slab Rates (2024 structure)
- Household Consumption Patterns (Regional studies)

---

## 💬 Feedback & Support

For questions, suggestions, or technical support:
- **Open an Issue**: [GitHub Issues]([https://github.com/yourusername/household-energy-forecasting/issues](https://github.com/Sakthipriyasridhar/household_energy_consumption_forecasting))
- **Contact Developer**: [Email](sakthipriyasridhar122003@gmail.com)

*Empowering households with AI-driven energy insights for sustainable living and significant cost savings through machine learning and practical optimization strategies.*
