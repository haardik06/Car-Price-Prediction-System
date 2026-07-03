# ==========================================
# preprocessing.py
# Data Loading and Data Cleaning
# ==========================================

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def load_dataset():
    df = pd.read_csv("dataset/cardekho_dataset.csv")
    return df

def clean_data(df):
    df_clean = df.copy()
    df_clean.drop("Unnamed: 0", axis=1, inplace=True)
    df_clean.drop("car_name", axis=1, inplace=True)

    return df_clean
