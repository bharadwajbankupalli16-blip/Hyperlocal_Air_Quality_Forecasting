import pandas as pd
import os

print("=" * 70)
print("BUILDING FINAL MULTIMODAL DATASET")
print("=" * 70)

# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

AQI_FILE = "data/raw/air_quality_data.csv"
ENV_FILE = "data/processed/real_environment_hyderabad.csv"

OUTPUT_FILE = "data/processed/final_multimodal_dataset.csv"


# ---------------------------------------------------------
# 1. Load original AQI + traffic dataset
# ---------------------------------------------------------

print("\n1. Loading AQI and traffic data...")

aqi = pd.read_csv(AQI_FILE)

aqi["DateTime"] = pd.to_datetime(
    aqi["DateTime"]
)

print(f"AQI records: {len(aqi)}")


# ---------------------------------------------------------
# 2. Load real environmental data
# ---------------------------------------------------------

print("\n2. Loading real environmental data...")

environment = pd.read_csv(ENV_FILE)

environment["DateTime"] = pd.to_datetime(
    environment["DateTime"]
)

print(
    f"Environmental records: {len(environment)}"
)


# ---------------------------------------------------------
# 3. Sort both datasets
# ---------------------------------------------------------

aqi = aqi.sort_values("DateTime")

environment = environment.sort_values(
    "DateTime"
)


# ---------------------------------------------------------
# 4. Merge datasets by nearest timestamp
# ---------------------------------------------------------

print("\n3. Matching timestamps...")

merged = pd.merge_asof(
    aqi,
    environment,
    on="DateTime",
    direction="nearest",
    tolerance=pd.Timedelta("1 hour"),
    suffixes=("_synthetic", "_real")
)


# ---------------------------------------------------------
# 5. Remove records without real environmental data
# ---------------------------------------------------------

before = len(merged)

merged = merged.dropna(
    subset=[
        "Real_PM2_5",
        "Real_PM10",
        "Real_Temperature_C"
    ]
)

after = len(merged)


# ---------------------------------------------------------
# 6. Add data-source information
# ---------------------------------------------------------

merged["AQI_Data_Source"] = "Synthetic prototype"

merged["Weather_Data_Source"] = "Open-Meteo"

merged["Air_Quality_Data_Source"] = "Open-Meteo Air Quality"


# ---------------------------------------------------------
# 7. Create output directory
# ---------------------------------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)


# ---------------------------------------------------------
# 8. Save final dataset
# ---------------------------------------------------------

merged.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# 9. Display information
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL MULTIMODAL DATASET CREATED")
print("=" * 70)

print(
    f"\nRecords before matching: {before}"
)

print(
    f"Records after matching : {after}"
)

print("\nDataset shape:")

print(merged.shape)


print("\nFinal columns:")

for column in merged.columns:
    print(f"  - {column}")


print("\nSample records:")

print(
    merged.head().to_string(index=False)
)


print("\nMissing values:")

print(
    merged.isnull().sum()
)


print("\nSaved to:")

print(OUTPUT_FILE)

print("=" * 70)