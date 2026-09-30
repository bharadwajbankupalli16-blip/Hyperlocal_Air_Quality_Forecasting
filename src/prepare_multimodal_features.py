import pandas as pd
import os

print("=" * 70)
print("PREPARING MULTIMODAL ML DATASET")
print("=" * 70)

INPUT_FILE = "data/processed/final_multimodal_dataset.csv"
OUTPUT_FILE = "data/processed/ml_multimodal_features.csv"

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

print("\n1. Loading final multimodal dataset...")

data = pd.read_csv(INPUT_FILE)

data["DateTime"] = pd.to_datetime(data["DateTime"])

print(f"Records loaded: {len(data)}")


# ---------------------------------------------------------
# 2. Sort by neighborhood and time
# ---------------------------------------------------------

data = data.sort_values(
    ["Neighborhood", "DateTime"]
).reset_index(drop=True)


# ---------------------------------------------------------
# 3. Time-based features
# ---------------------------------------------------------

print("\n2. Creating time features...")

data["Hour"] = data["DateTime"].dt.hour

data["DayOfWeek"] = data["DateTime"].dt.dayofweek

data["Month"] = data["DateTime"].dt.month

data["DayOfMonth"] = data["DateTime"].dt.day

data["IsWeekend"] = (
    data["DayOfWeek"] >= 5
).astype(int)

data["IsRushHour"] = (
    data["Hour"].isin([8, 9, 17, 18, 19])
).astype(int)


# ---------------------------------------------------------
# 4. Previous AQI features
# ---------------------------------------------------------

print("\n3. Creating historical AQI features...")

data["AQI_Lag_1"] = (
    data.groupby("Neighborhood")["AQI"]
    .shift(1)
)

data["AQI_Lag_3"] = (
    data.groupby("Neighborhood")["AQI"]
    .shift(3)
)

data["AQI_Lag_6"] = (
    data.groupby("Neighborhood")["AQI"]
    .shift(6)
)

data["AQI_Lag_24"] = (
    data.groupby("Neighborhood")["AQI"]
    .shift(24)
)


# ---------------------------------------------------------
# 5. Rolling AQI features
# ---------------------------------------------------------

data["AQI_Rolling_3H"] = (
    data.groupby("Neighborhood")["AQI"]
    .transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
)

data["AQI_Rolling_6H"] = (
    data.groupby("Neighborhood")["AQI"]
    .transform(
        lambda x: x.shift(1).rolling(6).mean()
    )
)

data["AQI_Rolling_24H"] = (
    data.groupby("Neighborhood")["AQI"]
    .transform(
        lambda x: x.shift(1).rolling(24).mean()
    )
)


# ---------------------------------------------------------
# 6. Traffic history
# ---------------------------------------------------------

print("\n4. Creating traffic features...")

data["Traffic_Lag_1"] = (
    data.groupby("Neighborhood")["Traffic_Volume"]
    .shift(1)
)

data["Traffic_Lag_24"] = (
    data.groupby("Neighborhood")["Traffic_Volume"]
    .shift(24)
)


# ---------------------------------------------------------
# 7. Create next-hour target
# ---------------------------------------------------------

print("\n5. Creating next-hour AQI target...")

data["Target_AQI"] = (
    data.groupby("Neighborhood")["AQI"]
    .shift(-1)
)


# ---------------------------------------------------------
# 8. Select ML features
# ---------------------------------------------------------

features = [
    # Existing environmental variables
    "Temperature_C",
    "Humidity_Percent",
    "Wind_Speed_mps",
    "Traffic_Volume",
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
    "Month",
    "DayOfMonth",
    "IsWeekend",
    "IsRushHour",

    # Historical AQI
    "AQI_Lag_1",
    "AQI_Lag_3",
    "AQI_Lag_6",
    "AQI_Lag_24",
    "AQI_Rolling_3H",
    "AQI_Rolling_6H",
    "AQI_Rolling_24H",

    # Historical traffic
    "Traffic_Lag_1",
    "Traffic_Lag_24"
]


# ---------------------------------------------------------
# 9. Keep only existing features
# ---------------------------------------------------------

available_features = [
    column
    for column in features
    if column in data.columns
]

missing_features = [
    column
    for column in features
    if column not in data.columns
]

if missing_features:

    print("\nWARNING: Missing features:")

    for column in missing_features:
        print(f"  - {column}")


# ---------------------------------------------------------
# 10. Create ML dataset
# ---------------------------------------------------------

ml_columns = [
    "DateTime",
    "Neighborhood"
] + available_features + [
    "AQI",
    "Target_AQI"
]

ml_data = data[ml_columns].copy()


# ---------------------------------------------------------
# 11. Remove rows with missing values
# ---------------------------------------------------------

print("\n6. Removing incomplete rows...")

before = len(ml_data)

ml_data = ml_data.dropna().reset_index(drop=True)

after = len(ml_data)

print(f"Rows before cleaning: {before}")
print(f"Rows after cleaning : {after}")


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
# 13. Display information
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ML MULTIMODAL DATASET CREATED")
print("=" * 70)

print(f"\nDataset shape: {ml_data.shape}")

print(f"\nNumber of features: {len(available_features)}")

print("\nFeatures:")

for feature in available_features:
    print(f"  - {feature}")

print("\nFirst 5 records:")

print(
    ml_data.head().to_string(index=False)
)

print("\nSaved to:")

print(OUTPUT_FILE)

print("=" * 70)