import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np
import os
from google.cloud import bigquery
import gcsfs

# GCS Bucket Details
BUCKET_NAME = "ml_bucket_p1"
FILE_NAME = "BostonHousing.csv"
GCS_PATH = f"gs://{BUCKET_NAME}/{FILE_NAME}"

print(f"Loading data from {GCS_PATH}")

# Load dataset from GCS
try:
    df = pd.read_csv(GCS_PATH, storage_options={"token": "google_default"})
except Exception as e:
    raise FileNotFoundError(f"Dataset not found at {GCS_PATH}. Error: {e}")

# Define features and target
X = df.drop(columns=["medv"])
y = df["medv"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Standardization
numeric_features = list(X_train.select_dtypes(include=['int64', 'float64']).columns)

preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numeric_features)]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save preprocessor and data
output_dir = "data/processed_sai"
os.makedirs(output_dir, exist_ok=True)

joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))

np.save(os.path.join(output_dir, "X_train.npy"), X_train_processed)
np.save(os.path.join(output_dir, "y_train.npy"), y_train.values)
np.save(os.path.join(output_dir, "X_test.npy"), X_test_processed)
np.save(os.path.join(output_dir, "y_test.npy"), y_test.values)

print("Data preprocessing complete.")

### === Upload Processed Data to BigQuery === ###

# BigQuery details
PROJECT_ID = "mlflow-0438"
DATASET_ID = "house_price"
TABLE_NAME = "price_table"

TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"

# Convert processed training data into a DataFrame for BigQuery
df_train = pd.DataFrame(X_train_processed, columns=numeric_features)
df_train["medv"] = y_train.values  # Add target column

# Initialize BigQuery client
client = bigquery.Client()

# Define BigQuery schema
schema = [
    bigquery.SchemaField(name, "FLOAT") for name in numeric_features
] + [bigquery.SchemaField("medv", "FLOAT")]

# Load data into BigQuery
job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")

try:
    job = client.load_table_from_dataframe(df_train, TABLE_ID, job_config=job_config)
    job.result()  # Wait for the job to complete
    print(f"Data successfully uploaded to BigQuery table {TABLE_ID}")
except Exception as e:
    print(f"Failed to upload data to BigQuery. Error: {e}")
