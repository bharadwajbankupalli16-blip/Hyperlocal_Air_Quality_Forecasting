import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 70)
print("MULTIMODAL AIR QUALITY FORECASTING MODEL")
print("=" * 70)


# ---------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------

INPUT_FILE = "data/processed/ml_multimodal_features.csv"

MODEL_FILE = "models/multimodal_aqi_random_forest.pkl"


# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------

print("\n1. Loading ML dataset...")

data = pd.read_csv(INPUT_FILE)

data["DateTime"] = pd.to_datetime(
    data["DateTime"]
)

print(f"Total records: {len(data)}")


# ---------------------------------------------------------
# 3. Define features
# ---------------------------------------------------------

features = [
    "Temperature_C",
    "Humidity_Percent",
    "Wind_Speed_mps",
    "Traffic_Volume",

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
    "Month",
    "DayOfMonth",
    "IsWeekend",
    "IsRushHour",

    "AQI_Lag_1",
    "AQI_Lag_3",
    "AQI_Lag_6",
    "AQI_Lag_24",

    "AQI_Rolling_3H",
    "AQI_Rolling_6H",
    "AQI_Rolling_24H",

    "Traffic_Lag_1",
    "Traffic_Lag_24"
]

target = "Target_AQI"


# ---------------------------------------------------------
# 4. Check features
# ---------------------------------------------------------

print("\n2. Checking features...")

missing_features = [
    feature
    for feature in features
    if feature not in data.columns
]

if missing_features:

    print("\nERROR: The following features are missing:")

    for feature in missing_features:
        print(f"  - {feature}")

    raise SystemExit


print(f"Features available: {len(features)}")


# ---------------------------------------------------------
# 5. Sort chronologically
# ---------------------------------------------------------

print("\n3. Sorting data chronologically...")

data = data.sort_values(
    "DateTime"
).reset_index(drop=True)


# ---------------------------------------------------------
# 6. Create X and y
# ---------------------------------------------------------

X = data[features]

y = data[target]


print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")


# ---------------------------------------------------------
# 7. Time-based train/test split
# ---------------------------------------------------------

print("\n4. Creating time-based train/test split...")

split_index = int(
    len(data) * 0.80
)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


print(f"Training records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")


# ---------------------------------------------------------
# 8. Train Random Forest
# ---------------------------------------------------------

print("\n5. Training Random Forest...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ---------------------------------------------------------
# 9. Make predictions
# ---------------------------------------------------------

print("\n6. Making test predictions...")

predictions = model.predict(
    X_test
)


# ---------------------------------------------------------
# 10. Calculate metrics
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
# 11. Display metrics
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"\nMAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ---------------------------------------------------------
# 12. Feature importance
# ---------------------------------------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTOP 15 IMPORTANT FEATURES")

print(
    importance.head(15).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 13. Save model
# ---------------------------------------------------------

print("\n7. Saving model...")

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)

print(f"\nModel saved to:")
print(MODEL_FILE)


# ---------------------------------------------------------
# 14. Save feature list
# ---------------------------------------------------------

FEATURE_FILE = "models/multimodal_features.txt"

with open(
    FEATURE_FILE,
    "w",
    encoding="utf-8"
) as file:

    for feature in features:
        file.write(feature + "\n")


print("\nFeature list saved to:")
print(FEATURE_FILE)


# ---------------------------------------------------------
# 15. Final message
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MULTIMODAL MODEL TRAINING COMPLETED")
print("=" * 70)