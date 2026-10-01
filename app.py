from pathlib import Path

import joblib
import numpy as np
import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODELS_DIR = Path(__file__).resolve().parent / "models"

st.set_page_config(
    page_title="Diabetes Risk Predictor · Odwa Manitshana",
    page_icon=str(ASSETS_DIR / "favicon.png"),
    layout="centered",
)


@st.cache_resource
def load_models():
    scaler_path = MODELS_DIR / "scaler.pkl"
    rf_path = MODELS_DIR / "random_forest.pkl"

    missing = [path.name for path in (scaler_path, rf_path) if not path.is_file()]
    if missing:
        return None, None, missing

    scaler = joblib.load(scaler_path)
    rf_model = joblib.load(rf_path)
    return scaler, rf_model, None


_scaler, rf_model, _missing_files = load_models()

if _missing_files:
    st.error(
        "Could not load required model files ("
        + ", ".join(_missing_files)
        + "). Ensure the models/ folder is present and try again."
    )
    st.stop()

st.title("Diabetes Risk Predictor")

st.write(
    "Enter patient information below. The model will estimate the probability "
    "that this patient has diabetes based on the Pima Indians Diabetes Dataset."
)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1,
        help="Number of times pregnant (count).",
    )
    glucose = st.number_input(
        "Glucose (mg/dL)",
        min_value=0,
        max_value=300,
        value=120,
        step=1,
        help="Plasma glucose concentration from a 2-hour oral glucose tolerance test.",
    )
    blood_pressure = st.number_input(
        "Blood pressure (mmHg)",
        min_value=0,
        max_value=200,
        value=70,
        step=1,
        help="Diastolic blood pressure.",
    )
    skin_thickness = st.number_input(
        "Skin thickness (mm)",
        min_value=0,
        max_value=100,
        value=20,
        step=1,
        help="Triceps skin fold thickness.",
    )

with col2:
    insulin = st.number_input(
        "Insulin (mIU/mL)",
        min_value=0,
        max_value=1000,
        value=80,
        step=1,
        help="2-hour serum insulin level.",
    )
    bmi = st.number_input(
        "BMI (kg/m²)",
        min_value=0.0,
        max_value=80.0,
        value=25.0,
        step=0.1,
        format="%.1f",
        help="Body mass index (weight in kg divided by height in m²).",
    )
    dpf = st.number_input(
        "Diabetes pedigree function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01,
        format="%.2f",
        help="A function that scores likelihood of diabetes based on family history.",
    )
    age = st.number_input(
        "Age (years)",
        min_value=0,
        max_value=120,
        value=33,
        step=1,
        help="Age in years.",
    )

st.caption(
    "This tool is for educational purposes only and is not a substitute for "
    "professional medical advice."
)

if st.button("Predict risk", type="primary"):
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

    st.caption("Model: Random Forest")
    probability_pct = prob_diabetes * 100.0
    st.metric(
        label="Diabetes probability",
        value=f"{probability_pct:.2f}%",
        delta=None,
    )

    if pred_class == 1:
        st.error("Model prediction: **High risk / likely to have diabetes.**")
    else:
        st.success("Model prediction: **Low risk / unlikely to have diabetes.**")

    st.caption(
        "This tool is for educational purposes only and is **not** a substitute for "
        "professional medical advice."
    )
