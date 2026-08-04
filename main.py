import streamlit as st
import pandas as pd
import joblib


model = joblib.load("heart_model.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's details below.")



age = st.number_input("Age", min_value=1, max_value=120, value=40)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)
gender = 1 if gender == "Male" else 0

cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=80,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120",
    [0, 1]
)

restecg = st.selectbox(
    "Resting ECG",
    [0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=60,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.number_input(
    "Old Peak",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)



input_data = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})


if hasattr(model, "feature_names_in_"):
    input_data = input_data[model.feature_names_in_]



if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
        st.write(f"Probability: **{probability[1]*100:.2f}%**")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.write(f"Probability: **{probability[0]*100:.2f}%**")