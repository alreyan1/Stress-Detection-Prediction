import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open("stress_model.pkl", "rb") as file:
    model = pickle.load(file)

# App title
st.set_page_config(page_title="Stress Detector", layout="centered")
st.title("🧠 Stress Level Detector")
st.write("Predict your stress status based on 5 health indicators.")

# Input fields
heart_rate = st.slider("Heart Rate (bpm)", 50, 150, 80)
blood_pressure = st.slider("Blood Pressure (mmHg)", 90, 180, 120)
spo2 = st.slider("SpO2 - Oxygen Saturation (%)", 85, 100, 98)
sleep_hours = st.slider("Sleep Hours (last 24h)", 0, 12, 6)
anxiety_level = st.slider("Anxiety Level (1 - 10)", 1, 10, 5)

# Prediction button
if st.button("Check Stress Level"):
    input_data = np.array([[heart_rate, blood_pressure, spo2, sleep_hours, anxiety_level]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ You are likely experiencing **Stress**. Please take care.")
    else:
        st.success("✅ You are **Not Stressed**. Keep up the good health!")

    st.caption("Note: This is a machine learning-based estimate. Consult a medical professional for advice.")
