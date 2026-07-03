# ==========================================
# model.py
# Model Training
# ==========================================

import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def split_data(data):
    X = data.drop("selling_price", axis=1)
    y = data["selling_price"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators = 100,
        random_state = 42
    )

    model.fit(X_train, y_train)
    return model

def save_model(model, label_encoders, X_train):

    # Create models folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save the trained model
    joblib.dump(model, "models/random_forest_model.pkl")

    # Save label encoders
    joblib.dump(label_encoders, "models/label_encoders.pkl")

    # Save feature names
    joblib.dump(list(X_train.columns), "models/feature_columns.pkl")

    print("\nModel saved successfully!")