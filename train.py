from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import numpy as np
from google.cloud import bigquery, storage
import pandas as pd
import argparse
import gcsfs

# GCP Configuration
PROJECT_ID = "mlflow-0438"
DATASET_ID = "house_price"
TABLE_NAME = "price_table"
BUCKET_NAME = "ml_bucket_p1"
MODEL_OUTPUT_PATH = f"gs://{BUCKET_NAME}/model"

# Initialize BigQuery Client
client = bigquery.Client()

# Load training data from BigQuery
QUERY = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`"
df_train = client.query(QUERY).to_dataframe()

# Extract features and target
X_train = df_train.drop(columns=["medv"])
y_train = df_train["medv"]

# Load preprocessor from GCS
fs = gcsfs.GCSFileSystem()
PREPROCESSOR_PATH = f"gs://{BUCKET_NAME}/preprocessor.joblib"

with fs.open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = joblib.load(f)

# Apply preprocessing
X_train_processed = preprocessor.transform(X_train)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train_processed, y_train)

# Save model locally
os.makedirs("model", exist_ok=True)
model_path = "model/boston_housing_model.joblib"
joblib.dump(model, model_path)

# Upload model to GCS
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob("model/boston_housing_model.joblib")
blob.upload_from_filename(model_path)

print(f"Model trained and uploaded to {MODEL_OUTPUT_PATH}/boston_housing_model.joblib")
