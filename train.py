from sklearn.ensemble import RandomForestRegressor
import joblib
import argparse
import os
import numpy as np
import gcsfs  # Import GCS File System

def train_and_save_model(X_train, y_train, model_output_path):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Ensure the model directory exists
    os.makedirs(model_output_path, exist_ok=True)

    # Save the trained model
    model_path = os.path.join(model_output_path, 'boston_housing_model.joblib')
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-output-path', dest='model_output_path',
                        default='gs://ml_bucket_p1/model', type=str, 
                        help='Path to save the trained model in GCS')
    args = parser.parse_args()

    # GCS Bucket Information
    BUCKET_NAME = "ml_bucket_p1"
    DATA_PATH = f"gs://{BUCKET_NAME}/processed_data"  # Updated to your GCS folder

    # Use GCS File System to Load Data
    fs = gcsfs.GCSFileSystem()

    with fs.open(os.path.join(DATA_PATH, "X_train.npy"), "rb") as f:
        X_train = np.load(f)

    with fs.open(os.path.join(DATA_PATH, "y_train.npy"), "rb") as f:
        y_train = np.load(f)

    # Train and save the model
    train_and_save_model(X_train, y_train, args.model_output_path)

    # Upload the model to GCS
    model_gcs_path = os.path.join(args.model_output_path, "boston_housing_model.joblib")
    fs.put("boston_housing_model.joblib", model_gcs_path)
    print(f"Model uploaded to {model_gcs_path}")
