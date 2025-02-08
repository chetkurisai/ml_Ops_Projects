from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import numpy as np
from google.cloud import bigquery, storage, aiplatform
import pandas as pd
import gcsfs

# GCP Configuration
PROJECT_ID = "mlflow-0438"
BUCKET_NAME = "ml_bucket_p1"
DATASET_ID = "house_price"
TABLE_NAME = "price_table"
MODEL_NAME = "boston_housing_model"
REGION = "us-central1"

MODEL_PATH = f"gs://{BUCKET_NAME}/{MODEL_NAME}.joblib"

# Initialize BigQuery Client
client = bigquery.Client()
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
local_model_path = "model/boston_housing_model.joblib"
joblib.dump(model, local_model_path)

# Upload model to GCS
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob(f"{MODEL_NAME}.joblib")
blob.upload_from_filename(local_model_path)

print(f"Model trained and uploaded to {MODEL_PATH}")
