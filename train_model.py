# ==========================================
# Car Price Prediction using Machine Learning
# Part 2 - Reading and Understanding Dataset
# ==========================================

# Import the required libraries

from utils.preprocessing import load_dataset, clean_data
from utils.visualization import (
    plot_selling_price_distribution,
    plot_selling_price_boxplot,
    plot_brand_distribution,
    plot_average_brand_price,
    plot_correlation_heatmap
)
from utils.feature_engineering import encode_features
from utils.model import (
    split_data,
    train_random_forest,
    save_model
)

from utils.evaluation import (
    evaluate_model,
    plot_feature_importance
)

df = load_dataset()
df_clean = clean_data(df)

print(df_clean.head())

print("\nShape of dataset:")
print(df_clean.shape)

print("\nColumn Name:")
print(df_clean.columns)

print("\nDataset information:")
print(df_clean.info())

print("\nMissing values:")
print(df_clean.isnull().sum())

print("\nDuplicate Rows:")
print(df_clean.duplicated().sum())

print("\nStatistical Summary:")
print(df_clean.describe())

# ==========================================
# Step 3 - Data Cleaning
# ==========================================

df_clean = df.copy();

df_clean.drop("Unnamed: 0", axis = 1, inplace = True)

print("\nColumns After Removing Unnamed: 0")
print(df_clean.columns)

# ==========================================
# Step 4 - Explore Categorical Features
# ==========================================

print("\nUnique values in Each Categorical Column\n")

categorical_columns = [
    "car_name",
    "brand",
    "model",
    "seller_type",
    "fuel_type",
    "transmission_type",
]

for column in categorical_columns:
    print(f"{column}: {df_clean[column].nunique()} unique values")

# ==========================================
# Step 5 - Remove Redundant Column
# ==========================================

df_clean.drop("car_name", axis = 1, inplace = True)

print("\nColumns After Removing car_name:\n")
print(df_clean.columns)

# ==========================================
# Step 6 - Distribution of Selling Price
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

plot_selling_price_distribution(df_clean)
# ==========================================
# Step 7 - Boxplot for Selling Price
# ==========================================

plot_selling_price_boxplot(df_clean)

# ==========================================
# Step 8 - Top 10 Car Brands
# ==========================================

plot_brand_distribution(df_clean)

# ==========================================
# Step 9 - Average Selling Price by Brand
# ==========================================

plot_average_brand_price(df_clean)

plot_correlation_heatmap(df_clean)

# Encode categorical features
df_encoded, label_encoders = encode_features(df_clean)

# Split data
X_train, X_test, y_train, y_test = split_data(df_encoded)

# Train the model
model = train_random_forest(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

print("\nRandom Forest Model Trained Successfully!")

predictions = evaluate_model(
    model,
    X_test,
    y_test
)

plot_feature_importance(model, X_train)

save_model(
    model,
    label_encoders,
    X_train
)