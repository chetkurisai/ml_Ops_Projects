import streamlit as st
import joblib
import pandas as pd
import numpy as np
from google.cloud import storage
import gcsfs

# GCP Configuration
BUCKET_NAME = "ml_bucket_p1"
MODEL_PATH = f"gs://{BUCKET_NAME}/model/boston_housing_model.joblib"
PREPROCESSOR_PATH = f"gs://{BUCKET_NAME}/preprocessor.joblib"

# Load model and preprocessor from GCS
fs = gcsfs.GCSFileSystem()

try:
    with fs.open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = joblib.load(f)

    with fs.open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)

    st.success("Model and preprocessor loaded successfully!")
except Exception as e:
    st.error(f"Error loading model/preprocessor: {e}")
    st.stop()

st.title("Boston Housing Price Prediction")

# Input form
feature_inputs = {}
columns = preprocessor.transformers_[0][2]  # Get column names

st.sidebar.header("Enter House Features")
for col in columns:
    feature_inputs[col] = st.sidebar.number_input(f"{col}", value=0.0)

if st.sidebar.button("Predict"):
    input_data = pd.DataFrame([feature_inputs])
    input_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_processed)[0]

    st.write(f"### Predicted House Price: ${prediction:.2f}")
