import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_FILE = "data/processed/real_ml_features.csv"
MODEL_FILE = "models/real_pm25_random_forest.pkl"

print("Loading real ML dataset...")

data = pd.read_csv(INPUT_FILE)

data["DateTime"] = pd.to_datetime(data["DateTime"], utc=True)

# Same feature list created in Step 51
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

# Sort chronologically
data = data.sort_values("DateTime").reset_index(drop=True)

X = data[FEATURES]
y = data[TARGET]

# ---------------------------------------
# Time-based train/test split
# ---------------------------------------

split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nDataset:")
print("Total records:", len(data))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining period:")
print(data["DateTime"].iloc[:split_index].min())
print(data["DateTime"].iloc[:split_index].max())

print("\nTesting period:")
print(data["DateTime"].iloc[split_index:].min())
print(data["DateTime"].iloc[split_index:].max())

# ---------------------------------------
# Train Random Forest
# ---------------------------------------

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------------------
# Predictions
# ---------------------------------------

predictions = model.predict(X_test)

# ---------------------------------------
# Evaluation
# ---------------------------------------

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n========================================")
print("REAL PM2.5 MODEL PERFORMANCE")
print("========================================")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ---------------------------------------
# Save model
# ---------------------------------------

Path(MODEL_FILE).parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(model, MODEL_FILE)

print(f"\nModel saved to:")
print(MODEL_FILE)

# ---------------------------------------
# Save test predictions
# ---------------------------------------

test_results = data.iloc[split_index:].copy()

test_results["Actual_PM2_5"] = y_test.values
test_results["Predicted_PM2_5"] = predictions

test_results["Prediction_Error"] = (
    test_results["Actual_PM2_5"]
    - test_results["Predicted_PM2_5"]
)

test_output = (
    "data/processed/"
    "real_pm25_test_predictions.csv"
)

test_results.to_csv(
    test_output,
    index=False
)

# ---------------------------------------
# Feature importance
# ---------------------------------------

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

importance_output = (
    "data/processed/"
    "real_pm25_feature_importance.csv"
)

importance.to_csv(
    importance_output,
    index=False
)

print("\nTop 15 important features:")

print(
    importance.head(15).to_string(index=False)
)

print("\nFiles created:")
print(test_output)
print(importance_output)