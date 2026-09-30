import pandas as pd

FILE = "data/raw/real_air_quality_hyderabad.csv"

print("=" * 70)
print("REAL AIR QUALITY DATA CHECK")
print("=" * 70)

# Load data
data = pd.read_csv(FILE)

print("\n1. Dataset shape")
print(data.shape)

print("\n2. Columns")
print(data.columns.tolist())

print("\n3. First 5 records")
print(data.head())

print("\n4. Missing values")
print(data.isnull().sum())

print("\n5. Data types")
print(data.dtypes)

print("\n6. Date range")
print("Start:", data["DateTime"].min())
print("End  :", data["DateTime"].max())

print("\n7. Pollution statistics")

pollution_columns = [
    "Real_PM2_5",
    "Real_PM10",
    "Real_CO",
    "Real_NO2",
    "Real_O3",
    "Real_SO2"
]

print(data[pollution_columns].describe())

print("\n" + "=" * 70)
print("DATA CHECK COMPLETED")
print("=" * 70)