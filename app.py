from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("dataset/cardekho_dataset.csv")

# ==========================================
# Load Trained Files
# ==========================================

model = joblib.load("models/random_forest_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


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

    # Read data from form
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

    # Ensure correct column order
    input_data = input_data[feature_columns]

    # Predict
    prediction = model.predict(input_data)[0]

    prediction = round(prediction, 2)

    return render_template(
        "result.html",
        prediction=prediction
    )


# ==========================================
# Run App
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)