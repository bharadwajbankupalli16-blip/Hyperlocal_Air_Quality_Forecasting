import pandas as pd
from pathlib import Path

OPENAQ_FILE = "data/processed/openaq_hyderabad_hourly.csv"
WEATHER_FILE = "data/raw/historical_weather_hyderabad.csv"
OUTPUT_FILE = "data/processed/historical_openaq_weather_hyderabad.csv"

print("Loading OpenAQ data...")
openaq = pd.read_csv(OPENAQ_FILE)

print("Loading historical weather data...")
weather = pd.read_csv(WEATHER_FILE)

# Convert timestamps to UTC
openaq["Hour"] = pd.to_datetime(openaq["Hour"], utc=True)
weather["DateTime"] = pd.to_datetime(weather["DateTime"], utc=True)

# Remove invalid timestamps
openaq = openaq.dropna(subset=["Hour"])
weather = weather.dropna(subset=["DateTime"])

# Sort before merge_asof
openaq = openaq.sort_values("Hour")
weather = weather.sort_values("DateTime")

print("\nOpenAQ records:", len(openaq))
print("Weather records:", len(weather))

print("\nOpenAQ time range:")
print(openaq["Hour"].min())
print(openaq["Hour"].max())

print("\nWeather time range:")
print(weather["DateTime"].min())
print(weather["DateTime"].max())

# Merge each OpenAQ hourly observation with nearest weather observation
merged = pd.merge_asof(
    openaq,
    weather,
    left_on="Hour",
    right_on="DateTime",
    direction="nearest",
    tolerance=pd.Timedelta("1 hour")
)

# Remove rows where no weather observation was available
merged = merged.dropna(subset=["Real_Temperature_C"])

# Rename timestamp
merged.rename(columns={"Hour": "DateTime"}, inplace=True)

# Remove duplicate weather timestamp column
if "DateTime_y" in merged.columns:
    merged.drop(columns=["DateTime_y"], inplace=True)

# Add source information
merged["Pollution_Source"] = "OpenAQ"
merged["Weather_Source"] = "Open-Meteo Historical Archive"

# Save
output = Path(OUTPUT_FILE)
output.parent.mkdir(parents=True, exist_ok=True)

merged.to_csv(output, index=False)

print("\n========================================")
print("Historical multimodal dataset created!")
print("========================================")

print("Records:", len(merged))
print("Stations:", merged["Station_Name"].nunique())

print("\nColumns:")
print(merged.columns.tolist())

print("\nMissing values:")
print(
    merged[
        [
            "OpenAQ_PM2_5",
            "OpenAQ_PM10",
            "OpenAQ_NO2",
            "OpenAQ_O3",
            "Real_Temperature_C",
            "Real_Humidity_Percent",
            "Real_Wind_Speed_mps"
        ]
    ].isna().sum()
)

print("\nSample:")
print(merged.head(10).to_string(index=False))

print(f"\nSaved to: {OUTPUT_FILE}")