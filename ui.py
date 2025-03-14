import streamlit as st
import numpy as np
import joblib

# 🎨 Custom Streamlit UI
st.set_page_config(page_title="🌧️ Rainfall Prediction", layout="centered")

# 🎯 Load Model & Scaler
model = joblib.load("rain_prediction_model.joblib")
scaler = joblib.load("scaler.joblib")

# 🏠 UI Header
st.markdown(
    "<h1 style='text-align: center; color: #007BFF;'>🌧️ Rainfall Prediction App</h1>",
    unsafe_allow_html=True,
)

st.write("### Enter Weather Details Below ⬇️")

# 🌡️ Input Features with Side-by-Side Layout
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
    wind_speed = st.number_input("🌬️ Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    cloud_cover = st.number_input("☁️ Cloud Cover (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

with col2:
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
    pressure = st.number_input("🌀 Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1010.0, step=0.1)
    precipitation = st.number_input("🌧️ Precipitation (mm)", min_value=0.0, max_value=500.0, value=2.0, step=0.1)

# 🚀 Predict Button with Styling
st.markdown(
    "<style>div.stButton > button:first-child {background-color: #007BFF; color: white; font-size: 18px;}</style>",
    unsafe_allow_html=True,
)

if st.button("🔍 Predict Rainfall"):
    # 🧠 Scale the Input Features
    input_features = np.array([[temperature, humidity, wind_speed, pressure, cloud_cover, precipitation]])
    scaled_features = scaler.transform(input_features)

    # 📊 Get Prediction
    prediction = model.predict(scaled_features)[0]
    result = "🌧️ Yes, it will rain today!" if prediction == 1 else "🌤️ No, it won't rain today."

    # 🎉 Show Result with Styling
    st.success(result)
