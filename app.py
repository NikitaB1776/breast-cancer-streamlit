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
# Custom CSS
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
# Load Model
# ==============================
@st.cache_resource
def load_model():
    with open("breast_cancer_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ==============================
# Feature Names (FROM TRAINING)
# ==============================
FEATURES = [
    "ClumpThickness",
    "UniformityCellSize",
    "UniformityCellShape",
    "MarginalAdhesion",
    "SingleEpithelialCellSize",
    "BareNuclei",
    "BlandChromatin",
    "NormalNucleoli",
    "Mitoses"
]

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
st.sidebar.header("🔍 Choose Input Method")
option = st.sidebar.radio(
    "How do you want to give data?",
    ("Manual Input", "Upload CSV File")
)

# ==============================
# Manual Input
# ==============================
if option == "Manual Input":
    st.subheader("✍️ Enter Cell Features (1–10)")
    user_input = []

    for feature in FEATURES:
        value = st.number_input(feature, min_value=1, max_value=10, value=1)
        user_input.append(value)

    input_data = np.array(user_input).reshape(1, -1)

    if st.button("🔬 Predict"):
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            # RandomForestClassifier default: 2 classes, encoded as 2/4
            # 2 -> Benign, 4 -> Malignant
            label = "Malignant" if prediction == 4 else "Benign"
            color = "#ffe6e6" if label == "Malignant" else "#e6ffe6"
            text_color = "#cc0000" if label == "Malignant" else "#006600"

            st.markdown(
                f'<div class="result-box" style="background-color:{color};color:{text_color};">⚠️ {label} Tumor Detected</div>',
                unsafe_allow_html=True
            )
            st.write(f"### Prediction Confidence: **{max(probability)*100:.2f}%**")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ==============================
# CSV Upload
# ==============================
else:
    st.subheader("📂 Upload CSV File")
    uploaded_file = st.file_uploader("Upload CSV with 9 feature columns (same order as training)", type=["csv"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)

            # Check if all required features exist
            if list(data.columns) != FEATURES:
                st.error(f"⚠️ CSV must have columns in this exact order:\n{FEATURES}")
            else:
                st.write("📊 Uploaded Data Preview")
                st.dataframe(data.head())

                predictions = model.predict(data)
                probabilities = model.predict_proba(data)

                results = data.copy()
                results["Prediction"] = ["Malignant" if p == 4 else "Benign" for p in predictions]
                results["Confidence (%)"] = [round(max(prob)*100, 2) for prob in probabilities]

                st.write("🧾 Prediction Results")
                st.dataframe(results)

        except Exception as e:
            st.error(f"⚠️ CSV processing error: {e}")

# ==============================
# Disclaimer
# ==============================
st.write("---")
st.markdown("""
⚠️ **Medical Disclaimer**  
This application is for educational purposes only.  
Not a substitute for professional medical advice.
""")
