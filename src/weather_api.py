import requests
import pandas as pd
from pathlib import Path


# ============================================================
# REAL WEATHER DATA FROM OPEN-METEO
# ============================================================

print("=" * 60)
print("REAL WEATHER DATA")
print("=" * 60)


# ============================================================
# HYDERABAD COORDINATES
# ============================================================

LATITUDE = 17.3850
LONGITUDE = 78.4867


# ============================================================
# API URL
# ============================================================

url = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# API PARAMETERS
# ============================================================

params = {

    "latitude": LATITUDE,

    "longitude": LONGITUDE,

    "hourly": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "wind_speed_10m,"
        "precipitation"
    ),

    "forecast_days": 2,

    "timezone": "Asia/Kolkata"

}


# ============================================================
# REQUEST WEATHER DATA
# ============================================================

print("\nConnecting to weather API...")

response = requests.get(
    url,
    params=params,
    timeout=30
)


# ============================================================
# CHECK RESPONSE
# ============================================================

if response.status_code != 200:

    print(
        "Weather API request failed."
    )

    print(
        "Status code:",
        response.status_code
    )

    print(
        response.text
    )

    raise SystemExit


print(
    "Weather API connection successful!"
)


# ============================================================
# READ JSON
# ============================================================

weather = response.json()


# ============================================================
# EXTRACT HOURLY DATA
# ============================================================

hourly = weather["hourly"]


weather_data = pd.DataFrame({

    "DateTime":
        hourly["time"],

    "Temperature_C":
        hourly["temperature_2m"],

    "Humidity_Percent":
        hourly["relative_humidity_2m"],

    "Wind_Speed_kmh":
        hourly["wind_speed_10m"],

    "Precipitation_mm":
        hourly["precipitation"]

})


# ============================================================
# CONVERT DATETIME
# ============================================================

weather_data["DateTime"] = pd.to_datetime(
    weather_data["DateTime"]
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nLatest weather data:")

print(
    weather_data.head(10)
)


# ============================================================
# SAVE DATA
# ============================================================

project_root = Path(
    __file__
).resolve().parent.parent


output_folder = (
    project_root
    / "data"
    / "raw"
)


output_folder.mkdir(
    parents=True,
    exist_ok=True
)


output_file = (
    output_folder
    / "real_weather_hyderabad.csv"
)


weather_data.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("WEATHER DATA SUMMARY")
print("=" * 60)

print(
    f"\nRecords: {len(weather_data)}"
)

print(
    f"Temperature: "
    f"{weather_data['Temperature_C'].min():.1f}"
    f" to "
    f"{weather_data['Temperature_C'].max():.1f} °C"
)

print(
    f"Humidity: "
    f"{weather_data['Humidity_Percent'].min():.1f}"
    f"% to "
    f"{weather_data['Humidity_Percent'].max():.1f}%"
)

print(
    f"Wind: "
    f"{weather_data['Wind_Speed_kmh'].min():.1f}"
    f" to "
    f"{weather_data['Wind_Speed_kmh'].max():.1f} km/h"
)


print("\nSaved to:")

print(output_file)

print("\nSUCCESS!")