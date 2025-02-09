import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np
import os
import gcsfs

# Define GCS bucket and file path
BUCKET_NAME = "ml_bucket_p1"
FILE_NAME = "BostonHousing.csv"
GCS_PATH = f"gs://{BUCKET_NAME}/{FILE_NAME}"

print(f"Loading data from {GCS_PATH}")

# Initialize GCS filesystem
fs = gcsfs.GCSFileSystem()

# Load dataset from GCS
try:
    with fs.open(GCS_PATH, "rb") as f:
        df = pd.read_csv(f)
    print("Data loaded successfully from GCS.")
except Exception as e:
    print(f"Error loading data from GCS: {e}")
    raise

# Define features and target
X = df.drop(columns=["medv"])
y = df["medv"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Standardization
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numeric_features)]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save preprocessor and processed data locally
output_dir = "data/processed_sai"
os.makedirs(output_dir, exist_ok=True)

joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
np.save(os.path.join(output_dir, "X_train.npy"), X_train_processed)
np.save(os.path.join(output_dir, "y_train.npy"), y_train.values)
np.save(os.path.join(output_dir, "X_test.npy"), X_test_processed)
np.save(os.path.join(output_dir, "y_test.npy"), y_test.values)

print("Data preprocessing complete.")
