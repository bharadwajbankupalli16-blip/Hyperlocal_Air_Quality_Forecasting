import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 70)
print("EVALUATING CURRENT MULTIMODAL MODEL")
print("=" * 70)


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

DATA_FILE = (
    "data/processed/"
    "current_ml_features.csv"
)

MODEL_FILE = (
    "models/"
    "current_multimodal_aqi_model.pkl"
)

OUTPUT_FILE = (
    "data/processed/"
    "model_evaluation.csv"
)


# ---------------------------------------------------------
# Features
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


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("\n1. Loading dataset...")

data = pd.read_csv(DATA_FILE)

data["DateTime"] = pd.to_datetime(
    data["DateTime"]
)

data = data.sort_values(
    "DateTime"
).reset_index(drop=True)


print(
    f"Total records: {len(data)}"
)


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------

split_index = int(
    len(data) * 0.80
)

train = data.iloc[
    :split_index
]

test = data.iloc[
    split_index:
]


X_test = test[features]

y_test = test["Target_AQI"]


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("\n2. Loading trained model...")

model = joblib.load(
    MODEL_FILE
)


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

print("\n3. Generating test predictions...")

predictions = model.predict(
    X_test
)


# ---------------------------------------------------------
# Metrics
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
# Print metrics
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nTraining records: {len(train)}"
)

print(
    f"Testing records : {len(test)}"
)

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
# Feature importance
# ---------------------------------------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTOP 10 FEATURES")

print(
    importance.head(10).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Save evaluation metrics
# ---------------------------------------------------------

evaluation = pd.DataFrame({
    "Metric": [
        "Training Records",
        "Testing Records",
        "MAE",
        "RMSE",
        "R2"
    ],
    "Value": [
        len(train),
        len(test),
        mae,
        rmse,
        r2
    ]
})


evaluation.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------

prediction_file = (
    "data/processed/"
    "model_test_predictions.csv"
)

prediction_data = test[
    [
        "DateTime",
        "Neighborhood",
        "Target_AQI"
    ]
].copy()

prediction_data[
    "Predicted_AQI"
] = predictions


prediction_data.to_csv(
    prediction_file,
    index=False
)


# ---------------------------------------------------------
# Save feature importance
# ---------------------------------------------------------

importance_file = (
    "data/processed/"
    "feature_importance.csv"
)

importance.to_csv(
    importance_file,
    index=False
)


print("\nFiles created:")

print(
    "- data/processed/model_evaluation.csv"
)

print(
    "- data/processed/model_test_predictions.csv"
)

print(
    "- data/processed/feature_importance.csv"
)

print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETED")
print("=" * 70)