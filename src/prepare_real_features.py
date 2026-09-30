import pandas as pd
from pathlib import Path

INPUT_FILE = "data/processed/historical_multimodal_training_data.csv"
OUTPUT_FILE = "data/processed/real_ml_features.csv"

print("Loading historical multimodal data...")

data = pd.read_csv(INPUT_FILE)

data["DateTime"] = pd.to_datetime(data["DateTime"], utc=True)

# Sort by monitoring station and time
data = data.sort_values(
    ["Station_Name", "DateTime"]
).reset_index(drop=True)

# ---------------------------------
# Time-based features
# ---------------------------------

local_time = data["DateTime"].dt.tz_convert("Asia/Kolkata")

data["Hour"] = local_time.dt.hour
data["DayOfWeek"] = local_time.dt.dayofweek
data["Month"] = local_time.dt.month
data["IsWeekend"] = (data["DayOfWeek"] >= 5).astype(int)

data["IsRushHour"] = (
    (
        (data["Hour"] >= 8)
        & (data["Hour"] <= 10)
    )
    |
    (
        (data["Hour"] >= 17)
        & (data["Hour"] <= 20)
    )
).astype(int)

# ---------------------------------
# Pollution lag features
# ---------------------------------

group = data.groupby("Station_Name")

data["PM2_5_Lag_1"] = group["OpenAQ_PM2_5"].shift(1)
data["PM2_5_Lag_3"] = group["OpenAQ_PM2_5"].shift(3)
data["PM2_5_Lag_6"] = group["OpenAQ_PM2_5"].shift(6)
data["PM2_5_Lag_24"] = group["OpenAQ_PM2_5"].shift(24)

data["PM2_5_Rolling_3H"] = (
    group["OpenAQ_PM2_5"]
    .transform(lambda x: x.rolling(3).mean())
)

data["PM2_5_Rolling_6H"] = (
    group["OpenAQ_PM2_5"]
    .transform(lambda x: x.rolling(6).mean())
)

# ---------------------------------
# Other pollution lags
# ---------------------------------

data["PM10_Lag_1"] = group["OpenAQ_PM10"].shift(1)
data["NO2_Lag_1"] = group["OpenAQ_NO2"].shift(1)
data["O3_Lag_1"] = group["OpenAQ_O3"].shift(1)

# ---------------------------------
# Traffic lag features
# ---------------------------------

traffic_group = data.groupby("Station_Name")

data["Traffic_Lag_1"] = traffic_group["Traffic_Volume"].shift(1)
data["Traffic_Lag_3"] = traffic_group["Traffic_Volume"].shift(3)

# ---------------------------------
# Weather lag features
# ---------------------------------

data["Temperature_Lag_1"] = (
    group["Real_Temperature_C"].shift(1)
)

data["Humidity_Lag_1"] = (
    group["Real_Humidity_Percent"].shift(1)
)

data["Wind_Lag_1"] = (
    group["Real_Wind_Speed_mps"].shift(1)
)

# ---------------------------------
# Prediction target
# ---------------------------------

data["Target_PM2_5"] = (
    group["OpenAQ_PM2_5"].shift(-1)
)

# ---------------------------------
# Features used by ML model
# ---------------------------------

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

# Keep only rows where the target and required features exist
required_columns = FEATURES + ["Target_PM2_5"]

data = data.dropna(
    subset=required_columns
).reset_index(drop=True)

# ---------------------------------
# Save
# ---------------------------------

output = Path(OUTPUT_FILE)
output.parent.mkdir(parents=True, exist_ok=True)

data.to_csv(output, index=False)

# Save feature list
feature_file = Path(
    "models/real_pm25_features.txt"
)

feature_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(feature_file, "w") as f:
    for feature in FEATURES:
        f.write(feature + "\n")

print("\n========================================")
print("REAL ML FEATURES CREATED")
print("========================================")

print("Records:", len(data))
print("Stations:", data["Station_Name"].nunique())

print("\nTarget:")
print("Target_PM2_5 = next-hour OpenAQ PM2.5")

print("\nFeatures:", len(FEATURES))

print("\nFeature list:")
for i, feature in enumerate(FEATURES, 1):
    print(f"{i}. {feature}")

print("\nTarget statistics:")
print(data["Target_PM2_5"].describe())

print("\nSample:")
print(
    data[
        [
            "DateTime",
            "Station_Name",
            "OpenAQ_PM2_5",
            "Target_PM2_5",
            "Traffic_Volume",
            "Real_Temperature_C",
            "Real_Humidity_Percent"
        ]
    ].head(10).to_string(index=False)
)

print(f"\nSaved to: {OUTPUT_FILE}")