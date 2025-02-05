import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Define paths
model_path = r"D:\ml_prjects\Projects\Ml_ops_Proj1\model\boston_housing_model.joblib"
preprocessor_path = r"D:\ml_prjects\Projects\BostonHousing_prediction\preprocessor.joblib"

# Load model and preprocessor
if not os.path.exists(model_path):
    st.error("Model not found. Please train and save the model first.")
    st.stop()

if not os.path.exists(preprocessor_path):
    st.error("Preprocessor not found. Please preprocess the data first.")
    st.stop()

# Load the trained model and preprocessor
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)
st.success("Model and preprocessor loaded successfully!")

st.title("Boston Housing Price Prediction")

# Input form for user
feature_inputs = {}
columns = preprocessor.transformers_[0][2]  # Get column names

st.sidebar.header("Enter House Features")
for col in columns:
    feature_inputs[col] = st.sidebar.number_input(f"{col}", value=0.0)

# Prediction button
if st.sidebar.button("Predict"):
    input_data = pd.DataFrame([feature_inputs])
    input_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_processed)[0]

    st.write(f"### Predicted House Price: ${prediction:.2f}")
