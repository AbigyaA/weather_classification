import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model, scaler, and encoder
model = joblib.load('weather_classification_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Define Streamlit app
st.title("Weather Classification App 🌦️")
st.write("Predict the weather type based on input features!")

# Sidebar for user input
st.sidebar.header("Input Features")

# Define input fields
def user_input_features():
    season = st.sidebar.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
    cloud_cover = st.sidebar.selectbox("Cloud Cover", ["partly cloudy", "overcast", "cloudy", "clear"])
    location = st.sidebar.selectbox("Location", ["mountain", "inland", "coastal"])
    numerical_data = {
        "Temperature": st.sidebar.slider("Temperature (°C)", -30, 130, 20),
        "Humidity": st.sidebar.slider("Humidity (%)", 10, 120, 50),
        "Wind Speed": st.sidebar.slider("Wind Speed (km/h)", 0, 70, 10),
        "Precipitation (%)": st.sidebar.slider("Precipitation (%)", 0, 120, 20),
        "Atmospheric Pressure": st.sidebar.slider("Atmospheric Pressure (hPa)", 700, 1300, 1013),
        "UV Index": st.sidebar.slider("UV Index", 0, 20, 5),
        "Visibility (km)": st.sidebar.slider("Visibility (km)", 0, 30, 10),
    }
    data = {**numerical_data, "Season": season, "Cloud Cover": cloud_cover, "Location": location}
    return pd.DataFrame([data])

# Get user input
input_df = user_input_features()

# Preprocess input
st.subheader("Input Data")
st.write(input_df)

# One-hot encode categorical data
categorical_features = ['Cloud Cover', 'Season', 'Location']
encoded_features = encoder.transform(input_df[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Scale numerical data
numerical_features = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "Precipitation (%)",
    "Atmospheric Pressure",
    "UV Index",
    "Visibility (km)"
]
scaled_numerical = scaler.transform(input_df[numerical_features])
scaled_df = pd.DataFrame(scaled_numerical, columns=numerical_features)

# Combine scaled numerical and encoded categorical features
processed_input = pd.concat([scaled_df, encoded_df], axis=1)

# Make predictions
prediction = model.predict(processed_input)[0]
prediction_proba = model.predict_proba(processed_input)[0]

# Display the results
st.subheader("Prediction")
st.write(f"Predicted Weather Type: **{prediction}**")

st.subheader("Prediction Probabilities")
for idx, class_name in enumerate(model.classes_):
    st.write(f"{class_name}: {prediction_proba[idx]:.2%}")
