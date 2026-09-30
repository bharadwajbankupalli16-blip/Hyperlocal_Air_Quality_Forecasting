import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# TRAFFIC REDUCTION WHAT-IF SCENARIO
# ============================================================

print("=" * 70)
print("TRAFFIC REDUCTION WHAT-IF AQI SCENARIO")
print("=" * 70)


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

print("Model loaded successfully.")


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(data_file)

data["DateTime"] = pd.to_datetime(
    data["DateTime"]
)


# ============================================================
# MODEL FEATURES
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
# SELECT A RECENT RECORD
# ============================================================

sample = data.iloc[-1].copy()


# ============================================================
# FUNCTION FOR AQI PREDICTION
# ============================================================

def predict_aqi(record):

    input_data = pd.DataFrame(
        [record[features].values],
        columns=features
    )

    prediction = model.predict(
        input_data
    )[0]

    return prediction


# ============================================================
# BASELINE PREDICTION
# ============================================================

baseline_prediction = predict_aqi(sample)

original_traffic = sample["Traffic_Volume"]


# ============================================================
# TRAFFIC REDUCTION
# ============================================================

traffic_reduction = 20


# Calculate reduced traffic

reduced_traffic = (
    original_traffic
    * (1 - traffic_reduction / 100)
)


# Create modified scenario

scenario = sample.copy()

scenario["Traffic_Volume"] = reduced_traffic


# Also modify traffic lag features

scenario["Traffic_Lag_1"] = (
    scenario["Traffic_Lag_1"]
    * (1 - traffic_reduction / 100)
)

scenario["Traffic_Lag_24"] = (
    scenario["Traffic_Lag_24"]
    * (1 - traffic_reduction / 100)
)


# ============================================================
# SCENARIO PREDICTION
# ============================================================

scenario_prediction = predict_aqi(
    scenario
)


# ============================================================
# AQI IMPROVEMENT
# ============================================================

aqi_change = (
    baseline_prediction
    - scenario_prediction
)


if baseline_prediction != 0:

    improvement_percent = (
        aqi_change
        / baseline_prediction
    ) * 100

else:

    improvement_percent = 0


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SCENARIO RESULTS")
print("=" * 70)

print(
    f"\nNeighborhood       : "
    f"{sample['Neighborhood']}"
)

print(
    f"Date/Time          : "
    f"{sample['DateTime']}"
)

print(
    f"\nOriginal Traffic   : "
    f"{original_traffic:.0f}"
)

print(
    f"Traffic Reduction  : "
    f"{traffic_reduction}%"
)

print(
    f"New Traffic        : "
    f"{reduced_traffic:.0f}"
)


print(
    f"\nBaseline AQI       : "
    f"{baseline_prediction:.2f}"
)

print(
    f"Scenario AQI       : "
    f"{scenario_prediction:.2f}"
)

print(
    f"\nAQI Change         : "
    f"{aqi_change:.2f}"
)

print(
    f"AQI Improvement    : "
    f"{improvement_percent:.2f}%"
)


# ============================================================
# INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if aqi_change > 0:

    print(
        "Reduced traffic produced a lower "
        "predicted AQI in this scenario."
    )

elif aqi_change < 0:

    print(
        "The model predicted a higher AQI "
        "under the reduced-traffic scenario."
    )

else:

    print(
        "The predicted AQI remained unchanged."
    )


print("\nSUCCESS!")