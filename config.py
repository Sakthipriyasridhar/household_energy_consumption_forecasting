# Theme Configuration
THEME = {
    "primary": "#2E86AB",       # Blue - Energy/Technology
    "secondary": "#A23B72",     # Purple - Innovation
    "accent": "#F18F01",        # Orange - Action/Warning
    "success": "#73AB84",       # Green - Savings/Success
    "background": "#F5F7FA",    # Light background
    "card": "#FFFFFF",          # Card background
    "text": "#2D3748",          # Dark text
    "text_light": "#718096",    # Light text
}

# App Configuration
APP_CONFIG = {
    "name": "Energy Optimizer AI",
    "description": "ML-Powered Household Energy Management",
    "version": "2.0",
    "author": "Sakthipriya S L",
    "streamlit_cloud_url": "https://householdenergyconsumptionforecasting-tqvrrpxyo9pbxod6geyusf.streamlit.app"
}

# TNEB Rate Configuration (Tamil Nadu)
TNEB_RATES = {
    "slabs": [
        {"range": (0, 100), "rate": 0},
        {"range": (101, 200), "rate": 2.25},
        {"range": (201, 400), "rate": 4.50},
        {"range": (401, 500), "rate": 6.00},
        {"range": (501, 600), "rate": 8.00},
        {"range": (601, 800), "rate": 9.00},
        {"range": (801, float('inf')), "rate": 10.00}
    ],
    "fixed_charges": 50
}

# Common Appliances Database
APPLIANCES = {
    "Refrigerator": {"power_w": 150, "daily_hours": 24, "category": "Essential"},
    "Air Conditioner": {"power_w": 1500, "daily_hours": 8, "category": "Cooling"},
    "Fan": {"power_w": 75, "daily_hours": 12, "category": "Cooling"},
    "LED Lights": {"power_w": 10, "daily_hours": 6, "category": "Lighting"},
    "Television": {"power_w": 120, "daily_hours": 4, "category": "Entertainment"},
    "Washing Machine": {"power_w": 500, "daily_hours": 1, "category": "Laundry"},
    "Water Heater": {"power_w": 2000, "daily_hours": 1, "category": "Heating"},
    "Microwave": {"power_w": 1200, "daily_hours": 0.5, "category": "Kitchen"},
    "Laptop": {"power_w": 65, "daily_hours": 6, "category": "Office"},
    "Mobile Charger": {"power_w": 10, "daily_hours": 3, "category": "Charging"}
}
