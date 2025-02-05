import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np
import os



# Load dataset
df_path = r"D:\ml_prjects\Data Sets\BostonHousing.csv"

if not os.path.exists(df_path):
    raise FileNotFoundError(f"Dataset not found at {df_path}")

df = pd.read_csv(df_path)

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
output_dir = r"D:\ml_prjects\Projects\BostonHousing_prediction"
os.makedirs(output_dir, exist_ok=True)

joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))

np.save(os.path.join(output_dir, "X_train.npy"), X_train_processed)
np.save(os.path.join(output_dir, "y_train.npy"), y_train.values)
np.save(os.path.join(output_dir, "X_test.npy"), X_test_processed)
np.save(os.path.join(output_dir, "y_test.npy"), y_test.values)

print("Data preprocessing complete.")