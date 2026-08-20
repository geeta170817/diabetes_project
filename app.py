import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model


model = load_model("diabetes_model.keras")
scaler = joblib.load("scaler.pkl")


st.title("Diabetes Prediction App")
st.write("Enter patient details")


pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    value=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0.0,
    value=120.0
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0.0,
    value=70.0
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0.0,
    value=20.0
)

insulin = st.number_input(
    "Insulin",
    min_value=0.0,
    value=80.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    value=30
)


if st.button("Predict"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    input_scaled = scaler.transform(input_data)

    probability = model.predict(input_scaled)[0][0]

    st.write(
        "Probability:",
        round(float(probability), 3)
    )

    if probability >= 0.5:
        st.warning(
            "Prediction: Diabetes Positive"
        )
    else:
        st.success(
            "Prediction: Diabetes Negative"
        )