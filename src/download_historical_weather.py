import requests
import pandas as pd
from pathlib import Path

LATITUDE = 17.3850
LONGITUDE = 78.4867

START_DATE = "2025-01-01"
END_DATE = "2025-02-20"

URL = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "precipitation"
    ],
    "timezone": "UTC"
}

print("Downloading historical weather data...")

response = requests.get(URL, params=params, timeout=60)

print("HTTP Status:", response.status_code)

response.raise_for_status()

data = response.json()

hourly = data["hourly"]

weather = pd.DataFrame({
    "DateTime": pd.to_datetime(hourly["time"], utc=True),
    "Real_Temperature_C": hourly["temperature_2m"],
    "Real_Humidity_Percent": hourly["relative_humidity_2m"],
    "Real_Wind_Speed_kmh": hourly["wind_speed_10m"],
    "Real_Precipitation_mm": hourly["precipitation"]
})

# Convert km/h to m/s
weather["Real_Wind_Speed_mps"] = (
    weather["Real_Wind_Speed_kmh"] / 3.6
)

weather.drop(columns=["Real_Wind_Speed_kmh"], inplace=True)

output = Path("data/raw/historical_weather_hyderabad.csv")
output.parent.mkdir(parents=True, exist_ok=True)

weather.to_csv(output, index=False)

print("\nHistorical weather downloaded successfully!")
print("Records:", len(weather))
print("Time range:")
print(weather["DateTime"].min())
print(weather["DateTime"].max())

print("\nColumns:")
print(weather.columns.tolist())

print("\nSample:")
print(weather.head().to_string(index=False))

print(f"\nSaved to: {output}")