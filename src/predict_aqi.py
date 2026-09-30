import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# AQI PREDICTION PIPELINE
# ============================================================

print("=" * 60)
print("HYPERLOCAL AQI PREDICTION")
print("=" * 60)


# ============================================================
# PROJECT PATHS
# ============================================================

project_root = Path(__file__).resolve().parent.parent

model_file = (
    project_root
    / "models"
    / "aqi_random_forest.pkl"
)

data_file = (
    project_root
    / "data"
    / "processed"
    / "air_quality_features.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load(model_file)

print("Model loaded successfully!")


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(data_file)

data["DateTime"] = pd.to_datetime(
    data["DateTime"]
)


# ============================================================
# FEATURES
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


# ============================================================
# TAKE A REAL RECORD FROM THE DATASET
# ============================================================

sample = data.iloc[-1].copy()


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame(
    [sample[features].values],
    columns=features
)


# ============================================================
# MAKE PREDICTION
# ============================================================

prediction = model.predict(
    input_data
)[0]


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(
    f"\nNeighborhood : {sample['Neighborhood']}"
)

print(
    f"Date/Time    : {sample['DateTime']}"
)

print(
    f"Traffic      : {sample['Traffic_Volume']:.0f}"
)

print(
    f"Temperature  : {sample['Temperature_C']:.2f} °C"
)

print(
    f"Humidity     : {sample['Humidity_Percent']:.2f} %"
)

print(
    f"\nCurrent AQI  : {sample['AQI']:.2f}"
)

print(
    f"Predicted AQI (next hour): {prediction:.2f}"
)


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


category = get_aqi_category(prediction)

print(
    f"Predicted Category: {category}"
)

print("\nSUCCESS!")