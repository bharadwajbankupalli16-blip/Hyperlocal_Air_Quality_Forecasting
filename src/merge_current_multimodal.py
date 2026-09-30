
import pandas as pd
import os

print("=" * 70)
print("BUILDING CURRENT MULTIMODAL DATASET")
print("=" * 70)

# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

ENV_FILE = "data/processed/real_environment_hyderabad.csv"

TRAINING_FILE = (
    "data/processed/"
    "current_neighborhood_training_data.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "current_multimodal_dataset.csv"
)


# ---------------------------------------------------------
# 1. Load real environment data
# ---------------------------------------------------------

print("\n1. Loading real environmental data...")

environment = pd.read_csv(ENV_FILE)

environment["DateTime"] = pd.to_datetime(
    environment["DateTime"]
)

print(
    f"Environmental records: {len(environment)}"
)


# ---------------------------------------------------------
# 2. Load neighborhood data
# ---------------------------------------------------------

print("\n2. Loading neighborhood and traffic data...")

training = pd.read_csv(TRAINING_FILE)

training["DateTime"] = pd.to_datetime(
    training["DateTime"]
)

print(
    f"Neighborhood records: {len(training)}"
)


# ---------------------------------------------------------
# 3. Sort datasets
# ---------------------------------------------------------

environment = environment.sort_values(
    "DateTime"
)

training = training.sort_values(
    "DateTime"
)


# ---------------------------------------------------------
# 4. Merge by DateTime
# ---------------------------------------------------------

print("\n3. Matching environmental timestamps...")

merged = pd.merge(
    training,
    environment,
    on="DateTime",
    how="inner"
)


# ---------------------------------------------------------
# 5. Sort final dataset
# ---------------------------------------------------------

merged = merged.sort_values(
    [
        "Neighborhood",
        "DateTime"
    ]
).reset_index(drop=True)


# ---------------------------------------------------------
# 6. Add source information
# ---------------------------------------------------------

merged["Weather_Source"] = "Open-Meteo"

merged["Air_Quality_Source"] = (
    "Open-Meteo Air Quality"
)

merged["Traffic_Source"] = (
    "Synthetic prototype"
)

merged["AQI_Source"] = (
    "Synthetic prototype"
)


# ---------------------------------------------------------
# 7. Save
# ---------------------------------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)

merged.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# 8. Display results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CURRENT MULTIMODAL DATASET CREATED")
print("=" * 70)

print(
    f"\nFinal records: {len(merged)}"
)

print(
    f"Neighborhoods: "
    f"{merged['Neighborhood'].nunique()}"
)

print(
    f"Start time: "
    f"{merged['DateTime'].min()}"
)

print(
    f"End time: "
    f"{merged['DateTime'].max()}"
)

print("\nColumns:")

for column in merged.columns:
    print(f"  - {column}")


print("\nFirst 5 records:")

print(
    merged.head().to_string(
        index=False
    )
)


print("\nMissing values:")

print(
    merged.isnull().sum()
)


print("\nSaved to:")

print(OUTPUT_FILE)

print("=" * 70)