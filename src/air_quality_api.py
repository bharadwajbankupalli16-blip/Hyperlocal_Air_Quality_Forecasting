import requests
import pandas as pd
import os

print("=" * 70)
print("REAL AIR QUALITY DATA COLLECTION")
print("=" * 70)

# Hyderabad location
LATITUDE = 17.3850
LONGITUDE = 78.4867

# Open-Meteo Air Quality API
URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# API parameters
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "hourly": [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "ozone",
        "sulphur_dioxide",
        "aerosol_optical_depth"
    ],
    "forecast_days": 2,
    "timezone": "Asia/Kolkata"
}

print("\nRequesting real air-quality data...")

try:
    response = requests.get(
        URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    print("API request successful!")

except requests.exceptions.RequestException as e:
    print("\nERROR: Could not connect to air-quality API.")
    print(e)
    raise SystemExit


# Get hourly data
hourly = data["hourly"]


# Create DataFrame
df = pd.DataFrame({
    "DateTime": hourly["time"],
    "Real_PM2_5": hourly["pm2_5"],
    "Real_PM10": hourly["pm10"],
    "Real_CO": hourly["carbon_monoxide"],
    "Real_NO2": hourly["nitrogen_dioxide"],
    "Real_O3": hourly["ozone"],
    "Real_SO2": hourly["sulphur_dioxide"],
    "Aerosol_Optical_Depth": hourly["aerosol_optical_depth"]
})


# Convert DateTime
df["DateTime"] = pd.to_datetime(df["DateTime"])


# Add location information
df["Location"] = "Hyderabad"
df["Latitude"] = LATITUDE
df["Longitude"] = LONGITUDE


# Make sure output directory exists
os.makedirs(
    "data/raw",
    exist_ok=True
)


# Output file
output_file = "data/raw/real_air_quality_hyderabad.csv"


# Save CSV
df.to_csv(
    output_file,
    index=False
)


# Display results
print("\n" + "=" * 70)
print("AIR QUALITY DATA COLLECTION SUCCESSFUL")
print("=" * 70)

print(f"\nRecords collected: {len(df)}")

print("\nColumns:")

for column in df.columns:
    print(f"  - {column}")

print("\nFirst 5 records:")

print(df.head().to_string(index=False))

print("\nMissing values:")

print(df.isnull().sum())

print("\nSaved to:")

print(output_file)

print("=" * 70)