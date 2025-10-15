import streamlit as st
import numpy as np
import pandas as pd
import pickle
import random

# --------------------------
# Load trained model
# --------------------------
model_path = "best_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🎗️", layout="wide")
st.title("🎗️ Breast Cancer Prediction App")
st.markdown("""
This app predicts whether a breast tumor is **Benign** or **Malignant** 
based on features from a digitized image of a fine needle aspirate (FNA) of a breast mass.
""")

# --------------------------
# Feature names
# --------------------------
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

# --------------------------
# Generate random default values dynamically
# --------------------------
# Optional: you can define realistic ranges for each feature
default_values = [round(random.uniform(0.5, 30.0), 3) for _ in feature_names]

# --------------------------
# Input fields for prediction
# --------------------------
st.header("🔢 Enter or adjust features for prediction")
cols = st.columns(3)
inputs = []

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        value = st.number_input(
            f"{feature.replace('_', ' ').title()}",
            value=float(default_values[i]),
            format="%.5f",
            key=f"input_{i}"
        )
        inputs.append(value)

# --------------------------
# Prediction
# --------------------------
if st.button("🔍 Predict"):
    try:
        input_data = np.array(inputs).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0]
            confidence = np.max(prob) * 100

        result = "Malignant" if prediction == 1 else "Benign"

        st.subheader("🩺 Prediction Result")
        st.success(f"The tumor is predicted to be **{result}**.")

        if confidence:
            st.info(f"Model confidence: **{confidence:.2f}%**")

        # --------------------------
        # Display user input data
        # --------------------------
        st.subheader("📊 Input Data Table")
        input_df = pd.DataFrame([inputs], columns=feature_names)
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

        # --------------------------
        # Graph-wise analysis
        # --------------------------
        st.subheader("📈 Feature Analysis")
        st.bar_chart(input_df.T.rename(columns={0: "Value"}))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
