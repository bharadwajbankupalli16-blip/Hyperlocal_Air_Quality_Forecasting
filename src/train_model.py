import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# AQI FORECASTING MODEL
# ============================================================

print("=" * 70)
print("HYPERLOCAL AQI FORECASTING MODEL")
print("=" * 70)


# ============================================================
# PROJECT PATHS
# ============================================================

project_root = Path(__file__).resolve().parent.parent

input_file = (
    project_root
    / "data"
    / "processed"
    / "air_quality_features.csv"
)

model_folder = project_root / "models"

model_folder.mkdir(
    parents=True,
    exist_ok=True
)

model_file = model_folder / "aqi_random_forest.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading processed dataset...")

data = pd.read_csv(input_file)

data["DateTime"] = pd.to_datetime(
    data["DateTime"]
)

print(f"Total records: {len(data)}")


# ============================================================
# SORT DATA
# ============================================================

data = data.sort_values(
    "DateTime"
).reset_index(drop=True)


# ============================================================
# CREATE FUTURE TARGET
# ============================================================

# We want to predict the AQI one hour into the future.

data["Target_AQI"] = (
    data
    .groupby("Neighborhood")["AQI"]
    .shift(-1)
)


# Remove rows where future AQI is unavailable

data = data.dropna(
    subset=["Target_AQI"]
).reset_index(drop=True)


# ============================================================
# SELECT FEATURES
# ============================================================

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
    "Traffic_Lag_24",

    "Temperature_Lag_1",
    "Humidity_Lag_1",
    "Wind_Lag_1"
]


target = "Target_AQI"


# ============================================================
# CHECK FEATURES
# ============================================================

print("\nNumber of features:", len(features))

print("\nFeatures used:")

for feature in features:
    print(" -", feature)


# ============================================================
# CREATE X AND Y
# ============================================================

X = data[features]

y = data[target]


print("\nFeature matrix shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
# ============================================================

# IMPORTANT:
# For forecasting we should NOT randomly shuffle the data.
#
# Earlier data -> training
# Later data  -> testing

split_index = int(
    len(data) * 0.80
)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


print("\nTraining records:", len(X_train))

print("Testing records:", len(X_test))


# ============================================================
# TRAIN RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest model...")

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


# ============================================================
# MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test
)


# ============================================================
# EVALUATION
# ============================================================

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


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"\nMAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 70)
print("TOP 10 IMPORTANT FEATURES")
print("=" * 70)

print(
    importance.head(10).to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    model_file
)


print("\n" + "=" * 70)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 70)

print("\nModel location:")
print(model_file)

print("\nSUCCESS!")