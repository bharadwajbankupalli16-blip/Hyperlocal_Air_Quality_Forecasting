import pandas as pd
import os

print("=" * 70)
print("COMBINING REAL WEATHER + REAL AIR QUALITY DATA")
print("=" * 70)

WEATHER_FILE = "data/raw/real_weather_hyderabad.csv"
AIR_FILE = "data/raw/real_air_quality_hyderabad.csv"

OUTPUT_FILE = "data/processed/real_environment_hyderabad.csv"


# ---------------------------------------------------------
# 1. Load weather
# ---------------------------------------------------------

print("\n1. Loading weather data...")

weather = pd.read_csv(WEATHER_FILE)

weather["DateTime"] = pd.to_datetime(
    weather["DateTime"]
)

print(f"Weather records: {len(weather)}")


# ---------------------------------------------------------
# 2. Load air quality
# ---------------------------------------------------------

print("\n2. Loading air-quality data...")

air = pd.read_csv(AIR_FILE)

air["DateTime"] = pd.to_datetime(
    air["DateTime"]
)

print(f"Air-quality records: {len(air)}")


# ---------------------------------------------------------
# 3. Convert wind speed
# ---------------------------------------------------------

weather["Real_Wind_Speed_mps"] = (
    weather["Wind_Speed_kmh"] / 3.6
)


# ---------------------------------------------------------
# 4. Select weather columns
# ---------------------------------------------------------

weather = weather[
    [
        "DateTime",
        "Temperature_C",
        "Humidity_Percent",
        "Real_Wind_Speed_mps",
        "Precipitation_mm"
    ]
].copy()


weather = weather.rename(
    columns={
        "Temperature_C": "Real_Temperature_C",
        "Humidity_Percent": "Real_Humidity_Percent",
        "Precipitation_mm": "Real_Precipitation_mm"
    }
)


# ---------------------------------------------------------
# 5. Select air-quality columns
# ---------------------------------------------------------

air = air[
    [
        "DateTime",
        "Real_PM2_5",
        "Real_PM10",
        "Real_CO",
        "Real_NO2",
        "Real_O3",
        "Real_SO2",
        "Aerosol_Optical_Depth"
    ]
].copy()


# ---------------------------------------------------------
# 6. Sort
# ---------------------------------------------------------

weather = weather.sort_values("DateTime")

air = air.sort_values("DateTime")


# ---------------------------------------------------------
# 7. Merge
# ---------------------------------------------------------

print("\n3. Matching timestamps...")

merged = pd.merge_asof(
    weather,
    air,
    on="DateTime",
    direction="nearest",
    tolerance=pd.Timedelta("1 hour")
)


# ---------------------------------------------------------
# 8. Remove unmatched records
# ---------------------------------------------------------

merged = merged.dropna(
    subset=[
        "Real_PM2_5",
        "Real_PM10",
        "Real_NO2"
    ]
).reset_index(drop=True)


# ---------------------------------------------------------
# 9. Add location
# ---------------------------------------------------------

merged["Location"] = "Hyderabad"

merged["Latitude"] = 17.3850

merged["Longitude"] = 78.4867


# ---------------------------------------------------------
# 10. Save
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
# 11. Display
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("REAL ENVIRONMENT DATASET CREATED")
print("=" * 70)

print(f"\nFinal records: {len(merged)}")

print("\nColumns:")

for column in merged.columns:
    print(f"  - {column}")

print("\nSample:")

print(
    merged.head().to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 70)