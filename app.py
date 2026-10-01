import os

import joblib
import numpy as np
import streamlit as st

MODELS_DIR = "models"

scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")

# Loaded at startup so missing/corrupt scaler.pkl fails fast (used by other saved models).
_scaler = joblib.load(scaler_path)
rf_model = joblib.load(rf_path)

st.title("Diabetes Risk Predictor")

st.write(
    "Enter patient information below. The model will estimate the probability "
    "that this patient has diabetes based on the Pima Indians Diabetes Dataset."
)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0, step=0.1, format="%.1f")
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01,
        format="%.2f",
    )
    age = st.number_input("Age", min_value=0, max_value=120, value=33, step=1)

if st.button("Predict"):
    input_values = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age,
    ]

    x_input = np.array(input_values).reshape(1, -1)

    prob_diabetes = rf_model.predict_proba(x_input)[0, 1]
    pred_class = rf_model.predict(x_input)[0]

    st.subheader("Prediction")

    st.write(f"Estimated probability of diabetes: **{prob_diabetes:.2%}**")

    if pred_class == 1:
        st.error("Model prediction: **High risk / likely to have diabetes.**")
    else:
        st.success("Model prediction: **Low risk / unlikely to have diabetes.**")

    st.caption(
        "This tool is for educational purposes only and is **not** a substitute for professional medical advice."
    )
