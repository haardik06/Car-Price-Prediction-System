from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ==========================================
# Base Directory
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(
    os.path.join(BASE_DIR, "dataset", "cardekho_dataset.csv")
)

# ==========================================
# Load Trained Files
# ==========================================

model = joblib.load(
    os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
)

label_encoders = joblib.load(
    os.path.join(BASE_DIR, "models", "label_encoders.pkl")
)

feature_columns = joblib.load(
    os.path.join(BASE_DIR, "models", "feature_columns.pkl")
)

# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():

    brands = sorted(df["brand"].unique())
    models = sorted(df["model"].unique())

    return render_template(
        "index.html",
        brands=brands,
        models=models
    )

# ==========================================
# Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        brand = request.form["brand"]
        model_name = request.form["model"]

        vehicle_age = int(request.form["vehicle_age"])
        km_driven = int(request.form["km_driven"])

        seller_type = request.form["seller_type"]
        fuel_type = request.form["fuel_type"]
        transmission_type = request.form["transmission_type"]

        mileage = float(request.form["mileage"])
        engine = int(request.form["engine"])
        max_power = float(request.form["max_power"])
        seats = int(request.form["seats"])

        # Encode categorical values
        brand = label_encoders["brand"].transform([brand])[0]
        model_name = label_encoders["model"].transform([model_name])[0]
        seller_type = label_encoders["seller_type"].transform([seller_type])[0]
        fuel_type = label_encoders["fuel_type"].transform([fuel_type])[0]
        transmission_type = label_encoders["transmission_type"].transform([transmission_type])[0]

        # Create input dataframe
        input_data = pd.DataFrame([{

            "brand": brand,
            "model": model_name,
            "vehicle_age": vehicle_age,
            "km_driven": km_driven,
            "seller_type": seller_type,
            "fuel_type": fuel_type,
            "transmission_type": transmission_type,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats

        }])

        # Arrange columns correctly
        input_data = input_data[feature_columns]

        # Predict
        prediction = model.predict(input_data)[0]

        prediction = round(prediction, 2)

        return render_template(
            "result.html",
            prediction=prediction
        )

    except Exception as e:

        return f"Error: {e}"

# ==========================================
# Run Flask App
# ==========================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)