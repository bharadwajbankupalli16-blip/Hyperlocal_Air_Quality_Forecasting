import pandas as pd
import os

print("=" * 70)
print("MULTIMODAL WEATHER + AIR QUALITY DATA INTEGRATION")
print("=" * 70)

# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
AQI_FILE = "data/raw/air_quality_data.csv"
WEATHER_FILE = "data/raw/real_weather_hyderabad.csv"
OUTPUT_FILE = "data/processed/multimodal_air_quality.csv"

# ---------------------------------------------------------
# Load AQI data
# ---------------------------------------------------------
print("\nLoading AQI data...")

aqi = pd.read_csv(AQI_FILE)

aqi["DateTime"] = pd.to_datetime(aqi["DateTime"])

print(f"AQI records: {len(aqi)}")

# ---------------------------------------------------------
# Load real weather data
# ---------------------------------------------------------
print("\nLoading real weather data...")

weather = pd.read_csv(WEATHER_FILE)

weather["DateTime"] = pd.to_datetime(weather["DateTime"])

print(f"Weather records: {len(weather)}")

# ---------------------------------------------------------
# Convert weather wind speed
# ---------------------------------------------------------
if "Wind_Speed_kmh" in weather.columns:

    weather["Wind_Speed_mps_Real"] = (
        weather["Wind_Speed_kmh"] / 3.6
    )

# Rename real weather columns
weather = weather.rename(
    columns={
        "Temperature_C": "Temperature_C_Real",
        "Humidity_Percent": "Humidity_Percent_Real",
        "Wind_Speed_mps_Real": "Wind_Speed_mps_Real"
    }
)

# ---------------------------------------------------------
# Select required weather columns
# ---------------------------------------------------------
weather_columns = [
    "DateTime",
    "Temperature_C_Real",
    "Humidity_Percent_Real",
    "Wind_Speed_mps_Real",
    "Precipitation_mm"
]

weather = weather[weather_columns]

# ---------------------------------------------------------
# Sort both datasets
# ---------------------------------------------------------
aqi = aqi.sort_values("DateTime")
weather = weather.sort_values("DateTime")

# ---------------------------------------------------------
# Match weather with AQI timestamps
# ---------------------------------------------------------
print("\nMatching weather with AQI timestamps...")

merged = pd.merge_asof(
    aqi,
    weather,
    on="DateTime",
    direction="nearest",
    tolerance=pd.Timedelta("1 hour")
)

# ---------------------------------------------------------
# Count matched records
# ---------------------------------------------------------
matched = merged["Temperature_C_Real"].notna().sum()

print(f"Matched records: {matched}")
print(f"Unmatched records: {len(merged) - matched}")

# ---------------------------------------------------------
# Remove records without real weather
# ---------------------------------------------------------
merged = merged.dropna(
    subset=[
        "Temperature_C_Real",
        "Humidity_Percent_Real",
        "Wind_Speed_mps_Real"
    ]
)

# ---------------------------------------------------------
# Create final multimodal weather columns
# ---------------------------------------------------------
merged["Real_Temperature_C"] = merged["Temperature_C_Real"]

merged["Real_Humidity_Percent"] = merged[
    "Humidity_Percent_Real"
]

merged["Real_Wind_Speed_mps"] = merged[
    "Wind_Speed_mps_Real"
]

merged["Real_Precipitation_mm"] = merged[
    "Precipitation_mm"
]

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
os.makedirs(
    "data/processed",
    exist_ok=True
)

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("INTEGRATION SUCCESSFUL")
print("=" * 70)

print(f"\nFinal records: {len(merged)}")

print("\nFinal columns:")

for column in merged.columns:
    print(f"  - {column}")

print("\nReal weather sample:")

print(
    merged[
        [
            "DateTime",
            "Neighborhood",
            "Real_Temperature_C",
            "Real_Humidity_Percent",
            "Real_Wind_Speed_mps",
            "Real_Precipitation_mm"
        ]
    ].head()
)

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("=" * 70)