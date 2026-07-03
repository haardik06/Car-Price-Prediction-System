# ==========================================
# visualization.py
# Data Visualization Functions
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

def plot_selling_price_distribution(data):

    plt.figure(figsize=(10,6))

    sns.histplot(
        data["selling_price"],
        bins=30,
        kde=True
    )

    plt.title("Distribution of Selling Price")
    plt.xlabel("Selling Price")
    plt.ylabel("Number of Cars")
    plt.show()

def plot_selling_price_boxplot(data):

    plt.figure(figsize=(10, 5))

    sns.boxplot(x=data["selling_price"])

    plt.title("Boxplot of Selling Price")

    plt.xlabel("Selling Price")

    plt.show()

def plot_brand_distribution(data):

    plt.figure(figsize=(12, 6))

    sns.countplot(
        data=data,
        y="brand",
        order=data["brand"].value_counts().head(10).index
    )

    plt.title("Top 10 Most Common Car Brands")

    plt.xlabel("Number of Cars")

    plt.ylabel("Brand")

    plt.show()

def plot_average_brand_price(data):

    plt.figure(figsize=(14, 6))

    brand_price = (
        data.groupby("brand")["selling_price"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
    )

    sns.barplot(
        x=brand_price.values,
        y=brand_price.index
    )

    plt.title("Top 10 Brands by Average Selling Price")

    plt.xlabel("Average Selling Price")

    plt.ylabel("Brand")

    plt.show()

def plot_correlation_heatmap(data):

    plt.figure(figsize=(10, 8))

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=["number"])

    # Calculate correlation matrix
    correlation_matrix = numeric_data.corr()

    # Draw heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    plt.show()