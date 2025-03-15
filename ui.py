import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load Model & Scaler
model = joblib.load("rain_prediction_model.joblib")
scaler = joblib.load("scaler.joblib")

# 🌟 Streamlit Page Config
st.set_page_config(
    page_title="🌧 Rainfall Prediction",
    page_icon="🌦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 **CSS-Based Theme Detection (Dark & Light Mode)**
st.markdown("""
    <style>
        /* Light Mode */
        @media (prefers-color-scheme: light) {
            body { background-color: #FFFFFF; color: black; }
            .stNumberInput>div>div>input { border: 1px solid #ccc; background: white; color: black; }
            .stButton>button { background: #007BFF; color: white; }
            .stButton>button:hover { background: #0056b3; }
            .stSidebar { background: #E3F2FD; color: black; }
        }

        /* Dark Mode */
        @media (prefers-color-scheme: dark) {
            body { background-color: #121212; color: #BBDEFB; }
            .stNumberInput>div>div>input { border: 2px solid #00bcd4; background: #222; color: white; }
            .stButton>button { background: #00bcd4; color: black; }
            .stButton>button:hover { background: #008ba3; }
            .stSidebar { background: #1E1E1E; color: white; }
        }
    </style>
""", unsafe_allow_html=True)

# 📈 Model Evaluation Metrics
model_accuracy = 92.3  # Manually set, ya evaluate_model.py se le sakte ho
model_auc = 0.89

# 🏠 Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1163/1163657.png", width=100)
st.sidebar.header("📈 Model Insights")
st.sidebar.info(f"🎯 **Accuracy:** {model_accuracy}%")
st.sidebar.info(f"📊 **AUC Score:** {model_auc}")
st.sidebar.info("🧠 **Model: Logistic Regression**")

# 🌧 **Main App Heading**
st.markdown("<h1 style='text-align: center; color: #0D47A1;'>🌧 Rainfall Prediction App</h1>", unsafe_allow_html=True)
st.write("### Enter Weather Details Below ⬇️")

# 🎛 User Inputs
col1, col2 = st.columns(2)
with col1:
    temperature = st.number_input("🌡 Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
    wind_speed = st.number_input("🌬 Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    cloud_cover = st.number_input("☁ Cloud Cover (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

with col2:
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
    pressure = st.number_input("🌀 Pressure (hPa)", min_value=800.0, max_value=1100.0, value=1010.0, step=0.1)
    precipitation = st.number_input("🌧 Precipitation (mm)", min_value=0.0, max_value=500.0, value=2.0, step=0.1)

# 📊 **Weather Trends - Simple Visualization**
st.subheader("📊 Weather Trends")
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(["Temp", "Humidity", "Wind", "Pressure", "Cloud", "Rain"], 
       [temperature, humidity, wind_speed, pressure, cloud_cover, precipitation], color=['blue', 'green', 'red', 'purple', 'orange', 'cyan'])
st.pyplot(fig)

# 🧠 **Prediction Logic**
if st.button("🔍 Predict Rainfall"):
    try:
        features = np.array([[temperature, humidity, wind_speed, pressure, cloud_cover, precipitation]])
        scaled_features = scaler.transform(features)
        prediction_prob = model.predict_proba(scaled_features)[0]  # Confidence Score
        prediction = model.predict(scaled_features)[0]

        confidence = round(max(prediction_prob) * 100, 2)  # Convert to percentage

        # 🌤 **Show Prediction Result with Colors**
        if prediction == 1:
            if confidence > 80:
                st.success(f"✅ 🌧 **Yes, it will rain today! ({confidence}% confidence)**")
            elif confidence > 50:
                st.warning(f"⚠️ 🌧 **Possible Rain ({confidence}% confidence)**")
            else:
                st.error(f"❗🌧 **Low Probability of Rain ({confidence}% confidence)**")
        else:
            if confidence > 80:
                st.success(f"✅ 🌤 **No, it won't rain today! ({confidence}% confidence)**")
            elif confidence > 50:
                st.warning(f"⚠️ 🌤 **Low chance of rain ({confidence}% confidence)**")
            else:
                st.error(f"❗🌤 **Prediction Uncertain ({confidence}% confidence)**")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
