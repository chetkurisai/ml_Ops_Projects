from sklearn.ensemble import RandomForestRegressor
import joblib
import argparse
import os
import numpy as np

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
                        default=r'D:\ml_prjects\Projects\BostonHousing_prediction\Model', type=str, 
                        help='Path to save the trained model')
    args = parser.parse_args()

    # Load training data
    data_path = r'D:\ml_prjects\Projects\BostonHousing_prediction'
    X_train = np.load(os.path.join(data_path, "X_train.npy"))
    y_train = np.load(os.path.join(data_path, "y_train.npy"))

    # Train and save the model
    train_and_save_model(X_train, y_train, args.model_output_path)
