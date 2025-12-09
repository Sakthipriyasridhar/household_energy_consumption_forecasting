import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_forecast(historical_kwh, months_ahead, user_profile, include_seasonality=True):
    """Generate mock forecast (replace with actual ML model)"""
    
    # Base consumption with some randomness
    base_monthly = historical_kwh
    
    # Generate months
    months = []
    predictions = []
    lower_bound = []
    upper_bound = []
    
    current_date = datetime.now()
    
    for i in range(months_ahead):
        month_date = current_date + timedelta(days=30*i)
        month_name = month_date.strftime("%b %Y")
        months.append(month_name)
        
        # Base prediction with trend
        base_pred = base_monthly * (1 + 0.01 * i)  # Small upward trend
        
        # Add seasonality
        if include_seasonality:
            month_num = month_date.month
            # Higher in summer (Apr-Jun), lower in winter (Nov-Jan)
            if month_num in [4, 5, 6]:
                seasonal_factor = 1.25  # 25% higher in summer
            elif month_num in [11, 12, 1]:
                seasonal_factor = 0.85  # 15% lower in winter
            else:
                seasonal_factor = 1.0
            
            base_pred *= seasonal_factor
        
        # Add some randomness
        random_factor = 1 + random.uniform(-0.05, 0.05)
        final_pred = base_pred * random_factor
        
        predictions.append(final_pred)
        
        # Confidence bounds (85% confidence)
        lower_bound.append(final_pred * 0.85)
        upper_bound.append(final_pred * 1.15)
    
    # Find peak month
    peak_idx = np.argmax(predictions)
    
    # Seasonal decomposition
    seasonal_data = []
    for i, pred in enumerate(predictions):
        trend = base_monthly * (1 + 0.01 * i)
        seasonal = pred - trend
        residual = random.uniform(-10, 10)
        
        seasonal_data.append({
            "month": months[i],
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual
        })
    
    # Cost projection
    cost_projection = {
        "Summer (Apr-Jun)": sum(predictions[3:6]) * 8.5,  # Higher rates in summer
        "Monsoon (Jul-Sep)": sum(predictions[6:9]) * 7.5,
        "Winter (Oct-Mar)": sum(predictions[9:] + predictions[:3]) * 7.0
    }
    
    # AI Insights
    insights = [
        {
            "title": "Summer Consumption Spike",
            "description": f"Expected {predictions[peak_idx]/base_monthly:.0%} increase in {months[peak_idx]} due to AC usage",
            "impact": f"Additional ₹{predictions[peak_idx]*8 - base_monthly*8:.0f} cost"
        },
        {
            "title": "Off-Peak Optimization",
            "description": "Shift 20% of consumption to off-peak hours (10 PM - 6 AM)",
            "impact": "Save ₹800-1,200/month"
        },
        {
            "title": "Appliance Efficiency",
            "description": "Your refrigerator accounts for 18% of total consumption",
            "impact": "Upgrade to save ₹600/month"
        }
    ]
    
    return {
        "months": months,
        "predictions": predictions,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "monthly_avg_kwh": np.mean(predictions),
        "peak_month": {"month": months[peak_idx], "value": predictions[peak_idx]},
        "total_annual_kwh": sum(predictions[:12]),
        "accuracy_score": 0.87 + random.uniform(-0.02, 0.02),
        "seasonal_decomposition": pd.DataFrame(seasonal_data),
        "cost_projection": cost_projection,
        "insights": insights
    }
