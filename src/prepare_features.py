import pandas as pd
import numpy as np
from pathlib import Path


# ============================================
# AIR QUALITY FEATURE ENGINEERING
# ============================================

print("=" * 60)
print("AIR QUALITY FEATURE ENGINEERING")
print("=" * 60)


# --------------------------------------------
# PROJECT PATHS
# --------------------------------------------

project_root = Path(__file__).resolve().parent.parent

input_file = (
    project_root
    / "data"
    / "raw"
    / "air_quality_data.csv"
)

output_folder = (
    project_root
    / "data"
    / "processed"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    output_folder
    / "air_quality_features.csv"
)


# --------------------------------------------
# LOAD DATA
# --------------------------------------------

print("\nLoading dataset...")

data = pd.read_csv(input_file)

print(f"Original records: {len(data)}")


# --------------------------------------------
# DATETIME CONVERSION
# --------------------------------------------

data["DateTime"] = pd.to_datetime(
    data["DateTime"]
)


# --------------------------------------------
# SORT DATA
# --------------------------------------------

data = data.sort_values(
    ["Neighborhood", "DateTime"]
).reset_index(drop=True)


# ============================================
# TIME FEATURES
# ============================================

print("Creating time features...")


# Hour of day
data["Hour"] = data["DateTime"].dt.hour


# Day of week
# Monday = 0
# Sunday = 6
data["DayOfWeek"] = data["DateTime"].dt.dayofweek


# Month
data["Month"] = data["DateTime"].dt.month


# Day of month
data["DayOfMonth"] = data["DateTime"].dt.day


# Weekend indicator
data["IsWeekend"] = (
    data["DayOfWeek"] >= 5
).astype(int)


# ============================================
# RUSH HOUR FEATURE
# ============================================

data["IsRushHour"] = (
    (
        (data["Hour"] >= 7)
        & (data["Hour"] <= 10)
    )
    |
    (
        (data["Hour"] >= 17)
        & (data["Hour"] <= 21)
    )
).astype(int)


# ============================================
# AQI LAG FEATURES
# ============================================

print("Creating historical AQI features...")


# Previous hour AQI
data["AQI_Lag_1"] = (
    data
    .groupby("Neighborhood")["AQI"]
    .shift(1)
)


# Previous 3 hours AQI
data["AQI_Lag_3"] = (
    data
    .groupby("Neighborhood")["AQI"]
    .shift(3)
)


# Previous 6 hours AQI
data["AQI_Lag_6"] = (
    data
    .groupby("Neighborhood")["AQI"]
    .shift(6)
)


# Previous 24 hours AQI
data["AQI_Lag_24"] = (
    data
    .groupby("Neighborhood")["AQI"]
    .shift(24)
)


# ============================================
# ROLLING AQI FEATURES
# ============================================

print("Creating rolling averages...")


# 3-hour rolling average
data["AQI_Rolling_3H"] = (
    data
    .groupby("Neighborhood")["AQI"]
    .transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
)


# 6-hour rolling average
data["AQI_Rolling_6H"] = (
    data
    .groupby("Neighborhood")["AQI"]
    .transform(
        lambda x: x.shift(1).rolling(6).mean()
    )
)


# 24-hour rolling average
data["AQI_Rolling_24H"] = (
    data
    .groupby("Neighborhood")["AQI"]
    .transform(
        lambda x: x.shift(1).rolling(24).mean()
    )
)


# ============================================
# TRAFFIC LAG
# ============================================

print("Creating traffic history features...")


data["Traffic_Lag_1"] = (
    data
    .groupby("Neighborhood")["Traffic_Volume"]
    .shift(1)
)


data["Traffic_Lag_24"] = (
    data
    .groupby("Neighborhood")["Traffic_Volume"]
    .shift(24)
)


# ============================================
# WEATHER HISTORY
# ============================================

data["Temperature_Lag_1"] = (
    data
    .groupby("Neighborhood")["Temperature_C"]
    .shift(1)
)


data["Humidity_Lag_1"] = (
    data
    .groupby("Neighborhood")["Humidity_Percent"]
    .shift(1)
)


data["Wind_Lag_1"] = (
    data
    .groupby("Neighborhood")["Wind_Speed_mps"]
    .shift(1)
)


# ============================================
# REMOVE ROWS WITH MISSING LAG VALUES
# ============================================

print("Removing incomplete historical records...")

before = len(data)

data = data.dropna().reset_index(drop=True)

after = len(data)

print(f"Records before cleaning: {before}")
print(f"Records after cleaning:  {after}")
print(f"Records removed:        {before - after}")


# ============================================
# CHECK MISSING VALUES
# ============================================

print("\nMissing values after processing:")

missing_values = data.isnull().sum()

print(
    missing_values[
        missing_values > 0
    ]
)


# ============================================
# SAVE PROCESSED DATA
# ============================================

data.to_csv(
    output_file,
    index=False
)


# ============================================
# DISPLAY INFORMATION
# ============================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print(f"\nFinal records: {len(data)}")
print(f"Final columns: {len(data.columns)}")

print("\nNew features:")

new_features = [
    "Hour",
    "DayOfWeek",
    "Month",
    "DayOfMonth",
    "IsWeekend",
    "IsRushHour",
    "AQI_Lag_1",
    "AQI_Lag_3",
    "AQI_Lag_6",
    "AQI_Lag_24",
    "AQI_Rolling_3H",
    "AQI_Rolling_6H",
    "AQI_Rolling_24H",
    "Traffic_Lag_1",
    "Traffic_Lag_24",
    "Temperature_Lag_1",
    "Humidity_Lag_1",
    "Wind_Lag_1"
]

for feature in new_features:
    print(f"  - {feature}")


print("\nFinal dataset preview:")
print(data.head())

print("\nSaved to:")
print(output_file)

print("\nSUCCESS!")