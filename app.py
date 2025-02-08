import json
import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import aiplatform

# GCP Configuration
PROJECT_ID = "mlflow-0438"
REGION = "us-central1"
ENDPOINT_ID = "boston-housing-endpoint"

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

# Function to send data to Vertex AI endpoint
def predict_from_vertex_ai(input_data):
    endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}")
    
    response = endpoint.predict(instances=input_data)
    
    return response.predictions

# Streamlit UI
st.title("Boston Housing Price Prediction")
st.write("Enter the house details to predict the price.")

# Create input fields
crim = st.number_input("Crime rate (CRIM)", min_value=0.0, step=0.1)
zn = st.number_input("Residential land zoned (ZN)", min_value=0.0, step=1.0)
indus = st.number_input("Non-retail business acres (INDUS)", min_value=0.0, step=0.1)
chas = st.selectbox("Charles River dummy variable (CHAS)", [0, 1])
nox = st.number_input("Nitric oxides concentration (NOX)", min_value=0.0, step=0.01)
rm = st.number_input("Average number of rooms (RM)", min_value=1.0, step=0.1)
age = st.number_input("Age of house (AGE)", min_value=0.0, step=1.0)
dis = st.number_input("Distance to employment centers (DIS)", min_value=0.0, step=0.1)
rad = st.number_input("Accessibility to highways (RAD)", min_value=1.0, step=1.0)
tax = st.number_input("Property tax rate (TAX)", min_value=0.0, step=1.0)
ptratio = st.number_input("Pupil-teacher ratio (PTRATIO)", min_value=0.0, step=0.1)
b = st.number_input("Proportion of Black residents (B)", min_value=0.0, step=1.0)
lstat = st.number_input("Lower status population percentage (LSTAT)", min_value=0.0, step=0.1)

# Format input data
input_data = [{
    "CRIM": crim, "ZN": zn, "INDUS": indus, "CHAS": chas, "NOX": nox,
    "RM": rm, "AGE": age, "DIS": dis, "RAD": rad, "TAX": tax,
    "PTRATIO": ptratio, "B": b, "LSTAT": lstat
}]

# Predict button
if st.button("Predict Price"):
    prediction = predict_from_vertex_ai(input_data)
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")
