import pandas as pd
import os


print("=" * 70)
print("MERGING REAL OPENAQ POLLUTION + WEATHER DATA")
print("=" * 70)


OPENAQ_FILE = (
    "data/processed/"
    "openaq_hyderabad_hourly.csv"
)

WEATHER_FILE = (
    "data/raw/"
    "real_weather_hyderabad.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "real_openaq_weather_hyderabad.csv"
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(OPENAQ_FILE):

    print(f"\nERROR: {OPENAQ_FILE} not found.")

    raise SystemExit(1)


if not os.path.exists(WEATHER_FILE):

    print(f"\nERROR: {WEATHER_FILE} not found.")

    raise SystemExit(1)


# ============================================================
# LOAD OPENAQ
# ============================================================

openaq = pd.read_csv(
    OPENAQ_FILE
)

print(
    f"\nOpenAQ records before cleaning: "
    f"{len(openaq)}"
)


# ============================================================
# OPENAQ DATETIME
# ============================================================

openaq["Hour"] = pd.to_datetime(
    openaq["Hour"],
    errors="coerce",
    utc=True
)


openaq = openaq.dropna(
    subset=["Hour"]
).copy()


# Force a consistent nanosecond UTC datetime type

openaq["Hour"] = (
    openaq["Hour"]
    .astype("datetime64[ns, UTC]")
)


print(
    f"OpenAQ records after cleaning: "
    f"{len(openaq)}"
)


# ============================================================
# STOP IF EMPTY
# ============================================================

if openaq.empty:

    print(
        "\nWARNING: OpenAQ hourly dataset contains "
        "zero valid records."
    )

    print(
        "\nThe merge cannot continue."
    )

    print(
        "\nFirst check:"
    )

    print(
        "data/processed/openaq_hyderabad_hourly.csv"
    )

    raise SystemExit(0)


# ============================================================
# LOAD WEATHER
# ============================================================

weather = pd.read_csv(
    WEATHER_FILE
)

print(
    f"\nWeather records: {len(weather)}"
)


# ============================================================
# WEATHER DATETIME
# ============================================================

weather["DateTime"] = pd.to_datetime(
    weather["DateTime"],
    errors="coerce"
)


weather = weather.dropna(
    subset=["DateTime"]
).copy()


# Convert India time → UTC

weather["DateTime"] = (
    weather["DateTime"]
    .dt.tz_localize(
        "Asia/Kolkata"
    )
    .dt.tz_convert("UTC")
)


# Force same datetime precision

weather["DateTime"] = (
    weather["DateTime"]
    .astype("datetime64[ns, UTC]")
)


# ============================================================
# RENAME WEATHER
# ============================================================

weather = weather.rename(
    columns={
        "Temperature_C":
            "Real_Temperature_C",

        "Humidity_Percent":
            "Real_Humidity_Percent",

        "Wind_Speed_mps":
            "Real_Wind_Speed_mps",

        "Precipitation_mm":
            "Real_Precipitation_mm"
    }
)


# ============================================================
# SELECT WEATHER COLUMNS
# ============================================================

weather_columns = [
    "DateTime",
    "Real_Temperature_C",
    "Real_Humidity_Percent",
    "Real_Wind_Speed_mps",
    "Real_Precipitation_mm"
]


weather = weather[
    [
        column
        for column in weather_columns
        if column in weather.columns
    ]
]


# ============================================================
# SORT
# ============================================================

openaq = openaq.sort_values(
    "Hour"
).reset_index(drop=True)


weather = weather.sort_values(
    "DateTime"
).reset_index(drop=True)


# ============================================================
# MERGE
# ============================================================

print(
    "\nMatching OpenAQ pollution with "
    "nearest weather observation..."
)


merged = pd.merge_asof(
    openaq,
    weather,
    left_on="Hour",
    right_on="DateTime",
    direction="nearest",
    tolerance=pd.Timedelta("1 hour")
)


# ============================================================
# REMOVE UNMATCHED WEATHER
# ============================================================

merged = merged.dropna(
    subset=["Real_Temperature_C"]
).copy()


# ============================================================
# SOURCE LABELS
# ============================================================

merged["Pollution_Source"] = (
    "OpenAQ real monitoring"
)

merged["Weather_Source"] = (
    "Open-Meteo"
)


# ============================================================
# SORT
# ============================================================

merged = merged.sort_values(
    [
        "Station_Name",
        "Hour"
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


merged.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    f"Merged records: {len(merged)}"
)

print(
    f"Stations: "
    f"{merged['Station_Name'].nunique()}"
)


print(
    "\nAvailable pollution columns:"
)


for column in [
    "OpenAQ_PM2_5",
    "OpenAQ_PM10",
    "OpenAQ_NO2",
    "OpenAQ_O3"
]:

    if column in merged.columns:

        print(
            f"- {column}"
        )


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


print("\n" + "=" * 70)
print("OPENAQ + WEATHER MERGE COMPLETED")
print("=" * 70)