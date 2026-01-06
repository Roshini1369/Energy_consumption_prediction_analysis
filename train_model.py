import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1️⃣ Load dataset
# =========================
df = pd.read_csv("data/Energy.csv")

# Remove unwanted unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("Columns after cleanup:", df.shape[1])

# =========================
# 2️⃣ Rename columns
# =========================
df.columns = [
    "Relative_Compactness",
    "Surface_Area",
    "Wall_Area",
    "Roof_Area",
    "Overall_Height",
    "Orientation",
    "Glazing_Area",
    "Glazing_Area_Distribution",
    "heating_load_kWh",
    "cooling_load_kWh"
]

# =========================
# 3️⃣ Handle missing values
# =========================
df = df.dropna(subset=["heating_load_kWh", "cooling_load_kWh"])

# =========================
# 4️⃣ Split features & targets
# =========================
X = df[
    [
        "Relative_Compactness",
        "Surface_Area",
        "Wall_Area",
        "Roof_Area",
        "Overall_Height",
        "Orientation",
        "Glazing_Area",
        "Glazing_Area_Distribution"
    ]
]

# 🔥 TARGETS (THIS IS THE LINE YOU ASKED ABOUT)
y = df[["heating_load_kWh", "cooling_load_kWh"]]

# =========================
# 5️⃣ Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6️⃣ Scale features
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 7️⃣ Train model
# =========================
model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
)

model.fit(X_train, y_train)

# =========================
# 8️⃣ Evaluate model
# =========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
rmsle = np.sqrt(
    mean_squared_error(np.log1p(y_test), np.log1p(y_pred))
)

print("\n📊 Model Evaluation Metrics")
print("MAE   :", round(mae, 3))
print("RMSE  :", round(rmse, 3))
print("RMSLE :", round(rmsle, 3))
print("R²    :", round(r2, 3))

# =========================
# 9️⃣ Save model & scaler
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\n✅ Model and scaler saved successfully")
