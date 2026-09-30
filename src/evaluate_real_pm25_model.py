import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

DATA_FILE = "data/processed/real_ml_features.csv"
MODEL_FILE = "models/real_pm25_random_forest.pkl"

METRICS_FILE = "data/processed/real_pm25_model_metrics.csv"
PREDICTIONS_FILE = "data/processed/real_pm25_model_predictions.csv"

FEATURES = [
    "OpenAQ_PM2_5",
    "OpenAQ_PM10",
    "OpenAQ_NO2",
    "OpenAQ_O3",

    "Real_Temperature_C",
    "Real_Humidity_Percent",
    "Real_Wind_Speed_mps",
    "Real_Precipitation_mm",

    "Traffic_Volume",
    "Traffic_Lag_1",
    "Traffic_Lag_3",

    "PM2_5_Lag_1",
    "PM2_5_Lag_3",
    "PM2_5_Lag_6",
    "PM2_5_Lag_24",

    "PM2_5_Rolling_3H",
    "PM2_5_Rolling_6H",

    "PM10_Lag_1",
    "NO2_Lag_1",
    "O3_Lag_1",

    "Temperature_Lag_1",
    "Humidity_Lag_1",
    "Wind_Lag_1",

    "Hour",
    "DayOfWeek",
    "Month",
    "IsWeekend",
    "IsRushHour"
]

TARGET = "Target_PM2_5"


print("Loading dataset...")

data = pd.read_csv(DATA_FILE)

data["DateTime"] = pd.to_datetime(
    data["DateTime"],
    utc=True
)

data = data.sort_values(
    "DateTime"
).reset_index(drop=True)


# ============================================================
# TIME-BASED TEST SPLIT
# ============================================================

split_index = int(len(data) * 0.80)

test_data = data.iloc[split_index:].copy()

X_test = test_data[FEATURES]
y_test = test_data[TARGET]


print("\nTest records:", len(test_data))

print(
    "Test period:",
    test_data["DateTime"].min(),
    "to",
    test_data["DateTime"].max()
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load(MODEL_FILE)

predictions = model.predict(X_test)


# ============================================================
# METRICS
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


print("\n========================================")
print("REAL PM2.5 MODEL EVALUATION")
print("========================================")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2"
    ],
    "Value": [
        mae,
        rmse,
        r2
    ]
})

Path(METRICS_FILE).parent.mkdir(
    parents=True,
    exist_ok=True
)

metrics.to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

test_data["Actual_PM2_5"] = y_test.values

test_data["Predicted_PM2_5"] = predictions

test_data["Prediction_Error"] = (
    test_data["Actual_PM2_5"]
    - test_data["Predicted_PM2_5"]
)

test_data.to_csv(
    PREDICTIONS_FILE,
    index=False
)


print("\nFiles created:")

print(METRICS_FILE)

print(PREDICTIONS_FILE)


# ============================================================
# ERROR SUMMARY
# ============================================================

print("\nPrediction error statistics:")

print(
    test_data["Prediction_Error"]
    .describe()
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\nSample predictions:")

print(
    test_data[
        [
            "DateTime",
            "Station_Name",
            "Actual_PM2_5",
            "Predicted_PM2_5",
            "Prediction_Error"
        ]
    ]
    .head(10)
    .to_string(index=False)
)