import pandas as pd
import os

print("=" * 70)
print("PREPARING CURRENT MULTIMODAL ML FEATURES")
print("=" * 70)

INPUT_FILE = (
    "data/processed/"
    "current_multimodal_dataset.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "current_ml_features.csv"
)


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

print("\n1. Loading current multimodal dataset...")

data = pd.read_csv(INPUT_FILE)

data["DateTime"] = pd.to_datetime(
    data["DateTime"]
)

print(f"Records loaded: {len(data)}")


# ---------------------------------------------------------
# 2. Sort by neighborhood and time
# ---------------------------------------------------------

data = data.sort_values(
    ["Neighborhood", "DateTime"]
).reset_index(drop=True)


# ---------------------------------------------------------
# 3. Time features
# ---------------------------------------------------------

print("\n2. Creating time features...")

data["Hour"] = data["DateTime"].dt.hour

data["DayOfWeek"] = (
    data["DateTime"].dt.dayofweek
)

data["IsWeekend"] = (
    data["DayOfWeek"] >= 5
).astype(int)

data["IsRushHour"] = (
    data["Hour"].isin(
        [8, 9, 17, 18, 19]
    )
).astype(int)


# ---------------------------------------------------------
# 4. Historical AQI features
# ---------------------------------------------------------

print("\n3. Creating AQI history features...")

group = data.groupby("Neighborhood")

data["AQI_Lag_1"] = (
    group["AQI"].shift(1)
)

data["AQI_Lag_3"] = (
    group["AQI"].shift(3)
)

data["AQI_Lag_6"] = (
    group["AQI"].shift(6)
)


# ---------------------------------------------------------
# 5. Rolling AQI
# ---------------------------------------------------------

data["AQI_Rolling_3H"] = (
    group["AQI"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(3)
        .mean()
    )
)


data["AQI_Rolling_6H"] = (
    group["AQI"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(6)
        .mean()
    )
)


# ---------------------------------------------------------
# 6. Traffic history
# ---------------------------------------------------------

print("\n4. Creating traffic history features...")

data["Traffic_Lag_1"] = (
    group["Traffic_Volume"].shift(1)
)


data["Traffic_Lag_3"] = (
    group["Traffic_Volume"].shift(3)
)


# ---------------------------------------------------------
# 7. Next-hour target
# ---------------------------------------------------------

print("\n5. Creating next-hour AQI target...")

data["Target_AQI"] = (
    group["AQI"].shift(-1)
)


# ---------------------------------------------------------
# 8. Select ML features
# ---------------------------------------------------------

features = [

    # Traffic
    "Traffic_Volume",
    "Traffic_Lag_1",
    "Traffic_Lag_3",

    # Existing pollution
    "PM2_5",
    "PM10",
    "NO2",
    "CO",
    "O3",

    # Real weather
    "Real_Temperature_C",
    "Real_Humidity_Percent",
    "Real_Wind_Speed_mps",
    "Real_Precipitation_mm",

    # Real air quality
    "Real_PM2_5",
    "Real_PM10",
    "Real_CO",
    "Real_NO2",
    "Real_O3",
    "Real_SO2",
    "Aerosol_Optical_Depth",

    # Time
    "Hour",
    "DayOfWeek",
    "IsWeekend",
    "IsRushHour",

    # AQI history
    "AQI_Lag_1",
    "AQI_Lag_3",
    "AQI_Lag_6",
    "AQI_Rolling_3H",
    "AQI_Rolling_6H"
]


# ---------------------------------------------------------
# 9. Check features
# ---------------------------------------------------------

missing = [
    feature
    for feature in features
    if feature not in data.columns
]

if missing:

    print("\nERROR: Missing features:")

    for feature in missing:
        print(" -", feature)

    raise SystemExit


# ---------------------------------------------------------
# 10. Create ML dataset
# ---------------------------------------------------------

ml_columns = [
    "DateTime",
    "Neighborhood"
] + features + [
    "AQI",
    "Target_AQI"
]

ml_data = data[ml_columns].copy()


# ---------------------------------------------------------
# 11. Remove rows with missing lag/target values
# ---------------------------------------------------------

print("\n6. Removing incomplete rows...")

before = len(ml_data)

ml_data = ml_data.dropna()

after = len(ml_data)

print(
    f"Rows before cleaning: {before}"
)

print(
    f"Rows after cleaning : {after}"
)


# ---------------------------------------------------------
# 12. Save
# ---------------------------------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)

ml_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# 13. Display
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CURRENT ML DATASET CREATED")
print("=" * 70)

print(
    f"\nDataset shape: {ml_data.shape}"
)

print(
    f"Number of features: {len(features)}"
)

print("\nFeatures:")

for feature in features:
    print(" -", feature)

print("\nFirst 5 records:")

print(
    ml_data.head().to_string(
        index=False
    )
)

print("\nSaved to:")

print(OUTPUT_FILE)

print("=" * 70)