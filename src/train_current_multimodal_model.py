import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 70)
print("CURRENT MULTIMODAL AQI FORECASTING MODEL")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

INPUT_FILE = (
    "data/processed/"
    "current_ml_features.csv"
)

MODEL_FILE = (
    "models/"
    "current_multimodal_aqi_model.pkl"
)

data = pd.read_csv(INPUT_FILE)

data["DateTime"] = pd.to_datetime(
    data["DateTime"]
)

print("\nTotal records:", len(data))


# ---------------------------------------------------------
# 2. Define features
# ---------------------------------------------------------

features = [
    "Traffic_Volume",
    "Traffic_Lag_1",
    "Traffic_Lag_3",

    "PM2_5",
    "PM10",
    "NO2",
    "CO",
    "O3",

    "Real_Temperature_C",
    "Real_Humidity_Percent",
    "Real_Wind_Speed_mps",
    "Real_Precipitation_mm",

    "Real_PM2_5",
    "Real_PM10",
    "Real_CO",
    "Real_NO2",
    "Real_O3",
    "Real_SO2",
    "Aerosol_Optical_Depth",

    "Hour",
    "DayOfWeek",
    "IsWeekend",
    "IsRushHour",

    "AQI_Lag_1",
    "AQI_Lag_3",
    "AQI_Lag_6",
    "AQI_Rolling_3H",
    "AQI_Rolling_6H"
]

target = "Target_AQI"


# ---------------------------------------------------------
# 3. Verify columns
# ---------------------------------------------------------

print("\nChecking features...")

missing = [
    column
    for column in features + [target]
    if column not in data.columns
]

if missing:

    print("\nERROR: Missing columns:")

    for column in missing:
        print(" -", column)

    raise SystemExit


print(
    f"Features available: {len(features)}"
)


# ---------------------------------------------------------
# 4. Sort chronologically
# ---------------------------------------------------------

data = data.sort_values(
    "DateTime"
).reset_index(drop=True)


# ---------------------------------------------------------
# 5. Create X and y
# ---------------------------------------------------------

X = data[features]

y = data[target]


# ---------------------------------------------------------
# 6. Time-based train/test split
# ---------------------------------------------------------

print("\nCreating time-based split...")

split_index = int(
    len(data) * 0.80
)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


print(
    f"Training records: {len(X_train)}"
)

print(
    f"Testing records : {len(X_test)}"
)


# ---------------------------------------------------------
# 7. Train Random Forest
# ---------------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ---------------------------------------------------------
# 8. Predict
# ---------------------------------------------------------

print("\nGenerating predictions...")

predictions = model.predict(
    X_test
)


# ---------------------------------------------------------
# 9. Calculate metrics
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


# ---------------------------------------------------------
# 10. Display metrics
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nMAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ---------------------------------------------------------
# 11. Feature importance
# ---------------------------------------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTOP 15 FEATURES")

print(
    importance.head(15).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 12. Save model
# ---------------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)

print("\nModel saved to:")

print(MODEL_FILE)


# ---------------------------------------------------------
# 13. Save feature list
# ---------------------------------------------------------

feature_file = (
    "models/"
    "current_multimodal_features.txt"
)

with open(
    feature_file,
    "w",
    encoding="utf-8"
) as file:

    for feature in features:
        file.write(
            feature + "\n"
        )


print("\nFeature list saved to:")

print(feature_file)


print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED")
print("=" * 70)