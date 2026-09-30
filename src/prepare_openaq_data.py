import pandas as pd
import os


print("=" * 70)
print("PREPARING REAL OPENAQ MONITORING DATA")
print("=" * 70)


INPUT_FILE = (
    "data/raw/"
    "openaq_hyderabad_measurements.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "clean_openaq_hyderabad.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(INPUT_FILE)

print(f"\nOriginal records: {len(data)}")


# ============================================================
# CONVERT TIMESTAMP
# ============================================================

data["DateTime_From"] = pd.to_datetime(
    data["DateTime_From"],
    errors="coerce",
    utc=True
)

data["DateTime_To"] = pd.to_datetime(
    data["DateTime_To"],
    errors="coerce",
    utc=True
)


# ============================================================
# REMOVE INVALID RECORDS
# ============================================================

data = data.dropna(
    subset=[
        "DateTime_From",
        "Value",
        "Parameter",
        "Latitude",
        "Longitude"
    ]
)


# ============================================================
# NUMERIC VALUES
# ============================================================

data["Value"] = pd.to_numeric(
    data["Value"],
    errors="coerce"
)

data = data.dropna(
    subset=["Value"]
)


# ============================================================
# REMOVE IMPOSSIBLE NEGATIVE VALUES
# ============================================================

data = data[
    data["Value"] >= 0
]


# ============================================================
# KEEP REQUIRED POLLUTANTS
# ============================================================

wanted = [
    "pm25",
    "pm10",
    "no2",
    "o3"
]

data = data[
    data["Parameter"].isin(wanted)
]


# ============================================================
# SORT
# ============================================================

data = data.sort_values(
    [
        "Station_Name",
        "Parameter",
        "DateTime_From"
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\nClean records:", len(data))

print("\nPollutants:")

print(
    data["Parameter"]
    .value_counts()
    .to_string()
)


print("\nStations:")

print(
    data["Station_Name"]
    .nunique()
)


print("\nDate range:")

print(
    "Start:",
    data["DateTime_From"].min()
)

print(
    "End  :",
    data["DateTime_From"].max()
)


print("\nMissing values:")

print(
    data.isnull()
    .sum()
    .to_string()
)


print("\nSaved:")

print(
    OUTPUT_FILE
)


print("\n" + "=" * 70)
print("OPENAQ DATA PREPARATION COMPLETED")
print("=" * 70)