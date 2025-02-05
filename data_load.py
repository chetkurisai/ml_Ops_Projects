import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np
import os


BUCKET_NAME = "ml_bucket_p1"
FILE_NAME = "BostonHousing.csv"
GCS_PATH = f"gs://{BUCKET_NAME}/{FILE_NAME}"

print(f"Loading data from {GCS_PATH}")

# Load dataset
# df_path = r"D:\ml_prjects\Data Sets\BostonHousing.csv"

if not os.path.exists(GCS_PATH):
    raise FileNotFoundError(f"Dataset not found at {GCS_PATH}")

df = pd.read_csv(GCS_PATH)

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


DATASET_ID = "house_price"
TABLE_NAME = "price_table"

TABLE_ID = f'{DATASET_ID}.{TABLE_NAME}'

print("Data preprocessing complete.")
