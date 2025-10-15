import streamlit as st
import numpy as np
import pickle

# --------------------------
# Load the trained model
# --------------------------
model_path = "best_model.pkl"  # Change if needed
with open(model_path, "rb") as file:
    model = pickle.load(file)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🎗️", layout="wide")

st.title("🎗️ Breast Cancer Prediction App")
st.markdown("""
This app uses a trained machine learning model to predict whether a breast tumor is **Benign** or **Malignant** 
based on input features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
""")

# --------------------------
# Input fields
# --------------------------
st.header("🔢 Enter the following features")

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

# Create input fields (grouped for readability)
cols = st.columns(3)
inputs = []
for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        value = st.number_input(f"{feature.replace('_', ' ').title()}", value=0.0, format="%.5f")
        inputs.append(value)

# --------------------------
# Predict button
# --------------------------
if st.button("🔍 Predict"):
    try:
        input_data = np.array(inputs).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        result = "Malignant" if prediction == 1 else "Benign"

        st.subheader("🩺 Prediction Result")
        st.success(f"The tumor is predicted to be **{result}**.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
