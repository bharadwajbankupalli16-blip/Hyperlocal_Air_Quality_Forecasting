import pandas as pd
import numpy as np
from pathlib import Path


# ============================================
# HYPERLOCAL AIR QUALITY DATASET GENERATOR
# ============================================

np.random.seed(42)

# Number of records
num_records = 3000

# Create hourly timestamps
dates = pd.date_range(
    start="2025-01-01",
    periods=num_records,
    freq="h"
)

# Neighborhoods
neighborhoods = [
    "Banjara Hills",
    "Madhapur",
    "Hitech City",
    "Kukatpally",
    "Gachibowli",
    "Secunderabad",
    "Ameerpet",
    "LB Nagar"
]

# Random neighborhood for each record
neighborhood = np.random.choice(
    neighborhoods,
    size=num_records
)

# --------------------------------------------
# WEATHER FEATURES
# --------------------------------------------

temperature = np.random.normal(
    loc=28,
    scale=5,
    size=num_records
)

humidity = np.random.normal(
    loc=60,
    scale=15,
    size=num_records
)

wind_speed = np.random.normal(
    loc=3.5,
    scale=1.5,
    size=num_records
)

# Keep values realistic
temperature = np.clip(temperature, 15, 42)
humidity = np.clip(humidity, 20, 95)
wind_speed = np.clip(wind_speed, 0.2, 12)


# --------------------------------------------
# TRAFFIC FEATURES
# --------------------------------------------

hour = dates.hour

# Higher traffic during morning/evening rush hours
rush_hour_effect = (
    ((hour >= 7) & (hour <= 10)) |
    ((hour >= 17) & (hour <= 21))
)

traffic_volume = np.random.normal(
    loc=500,
    scale=150,
    size=num_records
)

traffic_volume += rush_hour_effect * np.random.normal(
    350,
    80,
    num_records
)

traffic_volume = np.clip(
    traffic_volume,
    50,
    1500
)


# --------------------------------------------
# POLLUTION FEATURES
# --------------------------------------------

# PM2.5 is influenced by traffic, humidity and wind
pm25 = (
    20
    + traffic_volume * 0.035
    + humidity * 0.08
    - wind_speed * 2.5
    + np.random.normal(0, 8, num_records)
)

pm25 = np.clip(pm25, 5, 250)


# PM10
pm10 = (
    pm25 * 1.5
    + np.random.normal(0, 12, num_records)
)

pm10 = np.clip(pm10, 10, 400)


# NO2
no2 = (
    10
    + traffic_volume * 0.025
    + np.random.normal(0, 5, num_records)
)

no2 = np.clip(no2, 2, 150)


# CO
co = (
    0.2
    + traffic_volume * 0.0015
    + np.random.normal(0, 0.1, num_records)
)

co = np.clip(co, 0.1, 10)


# Ozone
o3 = (
    30
    + temperature * 1.2
    - humidity * 0.15
    + np.random.normal(0, 8, num_records)
)

o3 = np.clip(o3, 5, 180)


# --------------------------------------------
# AQI CALCULATION
# --------------------------------------------

# Simplified AQI proxy for our project prototype.
# Later we can replace this with the official
# pollutant-specific AQI calculation.

aqi = (
    pm25 * 0.55
    + pm10 * 0.20
    + no2 * 0.15
    + co * 5
    + o3 * 0.10
)

aqi += np.random.normal(
    0,
    5,
    num_records
)

aqi = np.clip(
    aqi,
    10,
    500
)


# --------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------

data = pd.DataFrame({
    "DateTime": dates,
    "Neighborhood": neighborhood,
    "Temperature_C": temperature.round(2),
    "Humidity_Percent": humidity.round(2),
    "Wind_Speed_mps": wind_speed.round(2),
    "Traffic_Volume": traffic_volume.round(0),
    "PM2_5": pm25.round(2),
    "PM10": pm10.round(2),
    "NO2": no2.round(2),
    "CO": co.round(2),
    "O3": o3.round(2),
    "AQI": aqi.round(2)
})


# --------------------------------------------
# SAVE DATASET
# --------------------------------------------

project_root = Path(__file__).resolve().parent.parent

output_folder = project_root / "data" / "raw"

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

output_file = output_folder / "air_quality_data.csv"

data.to_csv(
    output_file,
    index=False
)


# --------------------------------------------
# DISPLAY INFORMATION
# --------------------------------------------

print("=" * 60)
print("HYPERLOCAL AIR QUALITY DATASET CREATED")
print("=" * 60)

print(f"\nRecords: {len(data)}")
print(f"Columns: {len(data.columns)}")

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 records:")
print(data.head())

print("\nDataset saved to:")
print(output_file)

print("\nDataset shape:")
print(data.shape)

print("\nSUCCESS!")