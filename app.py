from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

# ================= LOAD MODEL =================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")

# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# ================= SINGLE PREDICTION =================
@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["Relative_Compactness"]),
        float(request.form["Surface_Area"]),
        float(request.form["Wall_Area"]),
        float(request.form["Roof_Area"]),
        float(request.form["Overall_Height"]),
        float(request.form["Orientation"]),
        float(request.form["Glazing_Area"]),
        float(request.form["Glazing_Area_Distribution"])
    ]

    X_scaled = scaler.transform([features])
    prediction = model.predict(X_scaled)[0]

    heating = round(prediction[0], 2)
    cooling = round(prediction[1], 2)
    total = round(heating + cooling, 2)

    return render_template(
        "result.html",
        heating=heating,
        cooling=cooling,
        total=total
    )

# ================= UPLOAD PAGE =================
@app.route("/upload")
def upload():
    return render_template("upload.html")

# ================= DATASET EVALUATION =================
@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        file = request.files["dataset"]
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        # ---- Map X1–X8, Y1–Y3 ----
        mapping = {
            "X1": "Relative_Compactness",
            "X2": "Surface_Area",
            "X3": "Wall_Area",
            "X4": "Roof_Area",
            "X5": "Overall_Height",
            "X6": "Orientation",
            "X7": "Glazing_Area",
            "X8": "Glazing_Area_Distribution",
            "Y1": "heating_load_kWh",
            "Y2": "cooling_load_kWh",
            "Y3": "total_energy_consumption"
        }
        df.rename(columns=mapping, inplace=True)

        features = [
            "Relative_Compactness", "Surface_Area", "Wall_Area",
            "Roof_Area", "Overall_Height", "Orientation",
            "Glazing_Area", "Glazing_Area_Distribution"
        ]

        targets = ["heating_load_kWh", "cooling_load_kWh"]

        # ---- Create total if missing ----
        if "total_energy_consumption" not in df.columns:
            df["total_energy_consumption"] = (
                df["heating_load_kWh"] + df["cooling_load_kWh"]
            )

        # ---- Validate columns ----
        required = features + targets
        missing = [c for c in required if c not in df.columns]
        if missing:
            return render_template(
                "evaluation.html",
                error=f"Missing columns: {', '.join(missing)}"
            )

        # ---- Handle NaN ----
        df[features] = df[features].fillna(df[features].mean())
        df = df.dropna(subset=targets)

        X = df[features]
        y_true = df[["heating_load_kWh", "cooling_load_kWh"]]

        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        # ---- Metrics (overall) ----
        r2 = round(r2_score(y_true, y_pred), 3)
        rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 3)
        mae = round(mean_absolute_error(y_true, y_pred), 3)

        return render_template(
            "evaluation.html",
            r2=r2,
            rmse=rmse,
            mae=mae
        )

    except Exception as e:
        return render_template("evaluation.html", error=str(e))

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
