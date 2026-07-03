# ==========================================
# evaluation.py
# Model Evaluation
# ==========================================

from sklearn.metrics import(
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print("\n=========== Model Evaluation Metrics ===============")

    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.2f}")
    print(f"MSE      : {mse:.2f}")
    print(f"RMSE     : {rmse:.2f}")

    return predictions 

def plot_feature_importance(model, X_train):

    # Get feature importance values
    importance = model.feature_importances_

    # Create DataFrame
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importance
    })

    # Sort values
    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    # Plot
    plt.figure(figsize=(10,6))

    sns.barplot(
        data=feature_importance,
        x="Importance",
        y="Feature"
    )

    plt.title("Feature Importance (Random Forest)")

    plt.xlabel("Importance")

    plt.ylabel("Features")

    plt.show()

