import pandas as pd
import os


print("=" * 70)
print("AGGREGATING OPENAQ DATA INTO HOURLY POLLUTION VALUES")
print("=" * 70)


INPUT_FILE = "data/raw/openaq_hyderabad_measurements.csv"

OUTPUT_FILE = "data/processed/openaq_hyderabad_hourly.csv"


# ============================================================
# LOAD
# ============================================================

data = pd.read_csv(INPUT_FILE)

print(f"\nRaw records: {len(data)}")

print("\nRaw columns:")
print(data.columns.tolist())


# ============================================================
# CLEAN PARAMETER
# ============================================================

data["Parameter"] = (
    data["Parameter"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# CLEAN VALUE
# ============================================================

data["Value"] = pd.to_numeric(
    data["Value"],
    errors="coerce"
)


# ============================================================
# CONVERT TIMESTAMP
# ============================================================

data["DateTime_From"] = pd.to_datetime(
    data["DateTime_From"],
    errors="coerce",
    utc=True
)


print(
    "\nValid timestamps:",
    data["DateTime_From"].notna().sum()
)

print(
    "Valid values:",
    data["Value"].notna().sum()
)


# ============================================================
# REMOVE INVALID
# ============================================================

data = data.dropna(
    subset=[
        "DateTime_From",
        "Value"
    ]
).copy()


data = data[
    data["Value"] >= 0
].copy()


# ============================================================
# KEEP POLLUTANTS
# ============================================================

wanted = [
    "pm25",
    "pm10",
    "no2",
    "o3"
]


data = data[
    data["Parameter"].isin(wanted)
].copy()


print(
    "\nRecords after pollutant filtering:",
    len(data)
)


print(
    "\nPollutants:"
)

print(
    data["Parameter"]
    .value_counts()
    .to_string()
)


# ============================================================
# CREATE HOURLY TIMESTAMP
# ============================================================

data["Hour"] = (
    data["DateTime_From"]
    .dt.floor("h")
)


# ============================================================
# AGGREGATE
# ============================================================

hourly = (
    data
    .groupby(
        [
            "Station_Name",
            "Location_ID",
            "Latitude",
            "Longitude",
            "Parameter",
            "Hour"
        ],
        as_index=False
    )["Value"]
    .mean()
)


print(
    "\nHourly long-format records:",
    len(hourly)
)


# ============================================================
# PIVOT
# ============================================================

hourly = hourly.pivot_table(
    index=[
        "Station_Name",
        "Location_ID",
        "Latitude",
        "Longitude",
        "Hour"
    ],
    columns="Parameter",
    values="Value",
    aggfunc="mean"
).reset_index()


hourly.columns.name = None


# ============================================================
# RENAME
# ============================================================

hourly = hourly.rename(
    columns={
        "pm25": "OpenAQ_PM2_5",
        "pm10": "OpenAQ_PM10",
        "no2": "OpenAQ_NO2",
        "o3": "OpenAQ_O3"
    }
)


# ============================================================
# SORT
# ============================================================

hourly = hourly.sort_values(
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


hourly.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    f"Hourly records created: {len(hourly)}"
)

print(
    f"Stations: {hourly['Station_Name'].nunique()}"
)


print("\nColumns:")

print(
    hourly.columns.tolist()
)


print("\nFirst 10 records:")

print(
    hourly.head(10).to_string(
        index=False
    )
)


print("\nSaved:")

print(
    OUTPUT_FILE
)


print("\n" + "=" * 70)
print("OPENAQ HOURLY AGGREGATION COMPLETED")
print("=" * 70)