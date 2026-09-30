import pandas as pd
import joblib
from pathlib import Path

DATA_FILE = "data/processed/real_ml_features.csv"
MODEL_FILE = "models/real_pm25_random_forest.pkl"
OUTPUT_FILE = "data/processed/real_pm25_traffic_scenario.csv"

TRAFFIC_REDUCTION = 20

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

print("Loading model and data...")

model = joblib.load(MODEL_FILE)
data = pd.read_csv(DATA_FILE)

data["DateTime"] = pd.to_datetime(
    data["DateTime"],
    utc=True
)

data = data.sort_values(
    ["Station_Name", "DateTime"]
)

# Latest record from every monitoring station
latest = (
    data.groupby("Station_Name", as_index=False)
    .tail(1)
    .copy()
)

latest = latest.reset_index(drop=True)

# ---------------------------------------
# Baseline prediction
# ---------------------------------------

latest["Baseline_PM2_5"] = model.predict(
    latest[FEATURES]
)

# ---------------------------------------
# Traffic reduction scenario
# ---------------------------------------

scenario = latest[FEATURES].copy()

reduction_factor = 1 - (
    TRAFFIC_REDUCTION / 100
)

scenario["Traffic_Volume"] *= reduction_factor
scenario["Traffic_Lag_1"] *= reduction_factor
scenario["Traffic_Lag_3"] *= reduction_factor

latest["Scenario_PM2_5"] = model.predict(
    scenario[FEATURES]
)

# ---------------------------------------
# Calculate change
# ---------------------------------------

latest["PM2_5_Change"] = (
    latest["Scenario_PM2_5"]
    - latest["Baseline_PM2_5"]
)

latest["Reduction_Percent"] = (
    (
        latest["Baseline_PM2_5"]
        - latest["Scenario_PM2_5"]
    )
    / latest["Baseline_PM2_5"]
    * 100
)

# ---------------------------------------
# Select output columns
# ---------------------------------------

results = latest[
    [
        "Station_Name",
        "DateTime",
        "OpenAQ_PM2_5",
        "Traffic_Volume",
        "Baseline_PM2_5",
        "Scenario_PM2_5",
        "PM2_5_Change",
        "Reduction_Percent"
    ]
].copy()

results["Traffic_Reduction_Percent"] = (
    TRAFFIC_REDUCTION
)

results["Traffic_Source"] = (
    "Synthetic prototype"
)

# ---------------------------------------
# Save
# ---------------------------------------

Path(OUTPUT_FILE).parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("REAL PM2.5 TRAFFIC SCENARIO")
print("========================================")

print(
    f"Traffic reduction: {TRAFFIC_REDUCTION}%"
)

print("\nResults:")

print(
    results.to_string(index=False)
)

print("\nAverage baseline PM2.5:",
      round(results["Baseline_PM2_5"].mean(), 3))

print("Average scenario PM2.5:",
      round(results["Scenario_PM2_5"].mean(), 3))

print(
    "Average predicted change:",
    round(results["PM2_5_Change"].mean(), 3)
)

print(
    "\nAverage predicted reduction:",
    round(results["Reduction_Percent"].mean(), 2),
    "%"
)

print(f"\nSaved to: {OUTPUT_FILE}")