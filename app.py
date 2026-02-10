import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

# ==============================
# Custom CSS (White Background)
# ==============================
st.markdown("""
<style>
body {
    background-color: white;
}
.main {
    background-color: white;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #333333;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666666;
    margin-bottom: 30px;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    font-size: 20px;
    text-align: center;
    font-weight: bold;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Load Model (ONLY MODEL)
# ==============================
@st.cache_resource
def load_model():
    with open("breast_cancer_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ==============================
# Title
# ==============================
st.markdown('<div class="title">🩺 Breast Cancer Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Machine Learning based early cancer prediction</div>',
    unsafe_allow_html=True
)

st.write("---")

# ==============================
# Sidebar
# ==============================
st.sidebar.header(" Choose Input Method")
option = st.sidebar.radio(
    "How do you want to give data?",
    ("Manual Input", "Upload CSV File")
)

# ==============================
# Manual Input Section
# ==============================
if option == "Manual Input":
    st.subheader(" Enter Cell Features")

    radius_mean = st.number_input("Radius Mean", min_value=0.0)
    texture_mean = st.number_input("Texture Mean", min_value=0.0)
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0)
    area_mean = st.number_input("Area Mean", min_value=0.0)
    smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0)

    input_data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean
    ]])

    if st.button(" Predict"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        if prediction == 1:
            st.markdown(
                '<div class="result-box" style="background-color:#ffe6e6;color:#cc0000;">⚠️ Malignant Tumor Detected</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box" style="background-color:#e6ffe6;color:#006600;">✅ Benign Tumor Detected</div>',
                unsafe_allow_html=True
            )

        st.write(f"### Prediction Confidence: **{max(probability)*100:.2f}%**")

# ==============================
# CSV Upload Section
# ==============================
else:
    st.subheader(" Upload CSV File")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            st.write(" Uploaded Data Preview")
            st.dataframe(data.head())

            predictions = model.predict(data)
            probabilities = model.predict_proba(data)

            results = data.copy()
            results["Prediction"] = ["Malignant" if p == 1 else "Benign" for p in predictions]
            results["Confidence (%)"] = [round(max(prob)*100, 2) for prob in probabilities]

            st.write(" Prediction Results")
            st.dataframe(results)

        except Exception as e:
            st.error("⚠️ Error processing file. Please check CSV format and feature order.")

# ==============================
# Disclaimer
# ==============================
st.write("---")
st.markdown("""
⚠️ **Medical Disclaimer**  
This application is for educational purposes only.  
It should **NOT** be used as a replacement for professional medical diagnosis.
""")

