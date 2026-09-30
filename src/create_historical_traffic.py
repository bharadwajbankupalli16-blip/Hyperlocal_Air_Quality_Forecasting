
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = "data/processed/historical_openaq_weather_hyderabad.csv"
OUTPUT_FILE = "data/processed/historical_multimodal_training_data.csv"

print("Loading historical OpenAQ + weather data...")

data = pd.read_csv(INPUT_FILE)

data["DateTime"] = pd.to_datetime(data["DateTime"], utc=True)

# Reproducible random generator
rng = np.random.default_rng(42)

# Extract hour in local India time
local_time = data["DateTime"].dt.tz_convert("Asia/Kolkata")
data["Hour"] = local_time.dt.hour
data["DayOfWeek"] = local_time.dt.dayofweek

# -----------------------------
# Prototype traffic generation
# -----------------------------

def traffic_pattern(hour):
    if 8 <= hour <= 10:
        return 1.35
    elif 17 <= hour <= 20:
        return 1.45
    elif 11 <= hour <= 16:
        return 1.05
    elif 21 <= hour <= 23:
        return 0.80
    else:
        return 0.55


data["Traffic_Pattern"] = data["Hour"].apply(traffic_pattern)

# Base traffic volume
base_traffic = 500

# Add daily variation and random noise
data["Traffic_Volume"] = (
    base_traffic
    * data["Traffic_Pattern"]
    * rng.uniform(0.85, 1.15, len(data))
).round().astype(int)

# Weekend adjustment
weekend_mask = data["DayOfWeek"] >= 5
data.loc[weekend_mask, "Traffic_Volume"] *= 0.85

data["Traffic_Volume"] = data["Traffic_Volume"].round().astype(int)

# Rush-hour indicator
data["IsRushHour"] = (
    ((data["Hour"] >= 8) & (data["Hour"] <= 10))
    |
    ((data["Hour"] >= 17) & (data["Hour"] <= 20))
).astype(int)

# Source label
data["Traffic_Source"] = "Synthetic prototype"

# Remove helper column
data.drop(columns=["Traffic_Pattern"], inplace=True)

# Save
output = Path(OUTPUT_FILE)
output.parent.mkdir(parents=True, exist_ok=True)

data.to_csv(output, index=False)

print("\n========================================")
print("Historical multimodal dataset created!")
print("========================================")

print("Records:", len(data))

print("\nTraffic statistics:")
print(data["Traffic_Volume"].describe())

print("\nRush-hour records:", data["IsRushHour"].sum())

print("\nTraffic source:")
print(data["Traffic_Source"].value_counts())

print("\nColumns:")
print(data.columns.tolist())

print("\nSample:")
print(
    data[
        [
            "DateTime",
            "Station_Name",
            "OpenAQ_PM2_5",
            "OpenAQ_PM10",
            "OpenAQ_NO2",
            "OpenAQ_O3",
            "Real_Temperature_C",
            "Real_Humidity_Percent",
            "Real_Wind_Speed_mps",
            "Traffic_Volume",
            "IsRushHour"
        ]
    ].head(10).to_string(index=False)
)

print(f"\nSaved to: {OUTPUT_FILE}")