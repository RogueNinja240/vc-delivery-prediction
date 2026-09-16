import streamlit as st
import requests
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Food Delivery ETA Predictor",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoint
# We will use localhost for now. When we Dockerize, we will update this.
API_URL = "http://localhost:8000"

# --- Header Section ---
st.title("🍔 Food Delivery ETA Predictor")
st.markdown("""
Welcome to the internal delivery routing dashboard. This tool utilizes a **Stacking Regressor** 
(Random Forest + LightGBM) to predict real-time delivery estimates based on live traffic, weather, and logistical data.
""")
st.divider()

# --- Application Tabs ---
tab1, tab2 = st.tabs(["🚀 Live Simulation (Auto-Pilot)", "🛠️ Manual Entry (Custom Order)"])

# ==========================================
# TAB 1: LIVE SIMULATION (Demo Endpoint)
# ==========================================
with tab1:
    st.header("Simulate a Live Delivery")
    st.write("Click the button below to sample a historical order, simulate it in real-time, and compare the model's prediction against the actual time it took.")
    
    if st.button("🔄 Simulate Random Order", type="primary"):
        with st.spinner("Fetching order details and computing ETA..."):
            try:
                response = requests.get(f"{API_URL}/predict/demo")
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display metrics beautifully
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Predicted ETA", f"{data['predicted_eta_minutes']} min")
                    
                    actual = data.get("actual_eta_minutes")
                    if actual:
                        col2.metric("Actual Delivery Time", f"{actual} min")
                        
                        # Color code the error margin (Green if under 5 mins, else red)
                        error = data['error_margin_minutes']
                        error_color = "normal" if error <= 5.0 else "inverse"
                        col3.metric("Margin of Error", f"{error} min", delta=f"{error}", delta_color=error_color)
                    
                    # Show the raw data that was processed
                    with st.expander("🔍 View Processed Order Features"):
                        st.json(data["simulated_order_features"])
                        
                else:
                    st.error(f"Failed to fetch simulation. API returned status code: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the API. Is your FastAPI backend running on port 8000?")

# ==========================================
# TAB 2: MANUAL ENTRY (Inference Endpoint)
# ==========================================
with tab2:
    st.header("Custom Order Prediction")
    st.write("Manually adjust environmental and logistical factors to see how the model adapts.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Driver Info")
            age = st.number_input("Driver Age", min_value=18, max_value=65, value=30)
            rating = st.slider("Driver Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
            vehicle = st.selectbox("Vehicle Type", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
            condition = st.selectbox("Vehicle Condition (0=Poor, 2=Excellent)", [0, 1, 2], index=2)
            
        with col2:
            st.subheader("Environment")
            weather = st.selectbox("Weather", ["conditions Sunny", "conditions Stormy", "conditions Sandstorms", "conditions Cloudy", "conditions Fog", "conditions Windy"])
            traffic = st.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
            city = st.selectbox("City Type", ["Metropolitian", "Urban", "Semi-Urban"])
            festival = st.selectbox("Is it a Festival?", ["No", "Yes"])

        with col3:
            st.subheader("Logistics")
            order_type = st.selectbox("Order Type", ["Snack", "Meal", "Drinks", "Buffet"])
            mult_deliveries = st.number_input("Multiple Deliveries", min_value=0, max_value=3, value=0)
            
            # Using some static dummy coordinates for the UI so the user doesn't have to guess lat/longs
            rest_lat, rest_long = 22.745049, 75.892471
            del_lat, del_long = 22.765049, 75.912471
            
        submit_button = st.form_submit_button("Predict ETA")
        
    if submit_button:
        # Build the payload matching our Pydantic Data model
        payload = {
            "ID": "0xUI_TEST",
            "Delivery_person_ID": "UI_DEMO_01",
            "Delivery_person_Age": str(age),
            "Delivery_person_Ratings": str(rating),
            "Restaurant_latitude": rest_lat,
            "Restaurant_longitude": rest_long,
            "Delivery_location_latitude": del_lat,
            "Delivery_location_longitude": del_long,
            "Order_Date": datetime.datetime.now().strftime("%d-%m-%Y"),
            "Time_Orderd": "12:00:00",
            "Time_Order_picked": "12:15:00",
            "Weatherconditions": weather,
            "Road_traffic_density": traffic,
            "Vehicle_condition": condition,
            "Type_of_order": order_type,
            "Type_of_vehicle": vehicle,
            "multiple_deliveries": str(mult_deliveries),
            "Festival": festival,
            "City": city
        }
        
        with st.spinner("Calculating custom ETA..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"🍔 Estimated Delivery Time: **{result['predicted_eta_minutes']} minutes**")
                else:
                    st.error("Error from API. Check your input formatting.")
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the API. Is your FastAPI backend running on port 8000?")