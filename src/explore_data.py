import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================
# HYPERLOCAL AIR QUALITY DATA ANALYSIS
# ============================================

# Find project directory
project_root = Path(__file__).resolve().parent.parent

# Dataset path
data_file = project_root / "data" / "raw" / "air_quality_data.csv"

# Load dataset
data = pd.read_csv(data_file)

# Convert DateTime column
data["DateTime"] = pd.to_datetime(data["DateTime"])

print("=" * 60)
print("AIR QUALITY DATA ANALYSIS")
print("=" * 60)

# --------------------------------------------
# BASIC INFORMATION
# --------------------------------------------

print("\n1. DATASET SHAPE")
print("-" * 40)
print(data.shape)

print("\n2. DATASET COLUMNS")
print("-" * 40)
print(data.columns.tolist())

print("\n3. FIRST 5 RECORDS")
print("-" * 40)
print(data.head())

# --------------------------------------------
# MISSING VALUES
# --------------------------------------------

print("\n4. MISSING VALUES")
print("-" * 40)
print(data.isnull().sum())

# --------------------------------------------
# STATISTICS
# --------------------------------------------

print("\n5. STATISTICAL SUMMARY")
print("-" * 40)
print(data.describe())

# --------------------------------------------
# NEIGHBORHOODS
# --------------------------------------------

print("\n6. NEIGHBORHOODS")
print("-" * 40)
print(data["Neighborhood"].value_counts())

# --------------------------------------------
# AVERAGE AQI
# --------------------------------------------

print("\n7. AVERAGE AQI")
print("-" * 40)

average_aqi = data["AQI"].mean()

print(f"Average AQI: {average_aqi:.2f}")

# --------------------------------------------
# MAXIMUM AQI
# --------------------------------------------

print("\n8. MAXIMUM AQI")
print("-" * 40)

max_aqi = data["AQI"].max()

print(f"Maximum AQI: {max_aqi:.2f}")

# --------------------------------------------
# MINIMUM AQI
# --------------------------------------------

print("\n9. MINIMUM AQI")
print("-" * 40)

min_aqi = data["AQI"].min()

print(f"Minimum AQI: {min_aqi:.2f}")


# ============================================
# CREATE VISUALIZATION FOLDER
# ============================================

visualization_folder = project_root / "assets" / "visualizations"

visualization_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================
# GRAPH 1 — AQI DISTRIBUTION
# ============================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data["AQI"],
    bins=30,
    kde=True
)

plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    visualization_folder / "aqi_distribution.png"
)

plt.show()

plt.close()


# ============================================
# GRAPH 2 — AQI OVER TIME
# ============================================

plt.figure(figsize=(12, 6))

plt.plot(
    data["DateTime"],
    data["AQI"],
    linewidth=1
)

plt.title("AQI Over Time")
plt.xlabel("Date")
plt.ylabel("AQI")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    visualization_folder / "aqi_over_time.png"
)

plt.show()

plt.close()


# ============================================
# GRAPH 3 — TRAFFIC VS AQI
# ============================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=data,
    x="Traffic_Volume",
    y="AQI",
    alpha=0.5
)

plt.title("Traffic Volume vs AQI")
plt.xlabel("Traffic Volume")
plt.ylabel("AQI")

plt.tight_layout()

plt.savefig(
    visualization_folder / "traffic_vs_aqi.png"
)

plt.show()

plt.close()


# ============================================
# GRAPH 4 — TEMPERATURE VS AQI
# ============================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=data,
    x="Temperature_C",
    y="AQI",
    alpha=0.5
)

plt.title("Temperature vs AQI")
plt.xlabel("Temperature (°C)")
plt.ylabel("AQI")

plt.tight_layout()

plt.savefig(
    visualization_folder / "temperature_vs_aqi.png"
)

plt.show()

plt.close()


# ============================================
# GRAPH 5 — AVERAGE AQI BY NEIGHBORHOOD
# ============================================

neighborhood_aqi = (
    data.groupby("Neighborhood")["AQI"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))

neighborhood_aqi.plot(
    kind="bar"
)

plt.title("Average AQI by Neighborhood")
plt.xlabel("Neighborhood")
plt.ylabel("Average AQI")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    visualization_folder / "aqi_by_neighborhood.png"
)

plt.show()

plt.close()


# ============================================
# CORRELATION ANALYSIS
# ============================================

numeric_columns = [
    "Temperature_C",
    "Humidity_Percent",
    "Wind_Speed_mps",
    "Traffic_Volume",
    "PM2_5",
    "PM10",
    "NO2",
    "CO",
    "O3",
    "AQI"
]

correlation = data[numeric_columns].corr()

print("\n10. CORRELATION WITH AQI")
print("-" * 40)

print(
    correlation["AQI"]
    .sort_values(ascending=False)
)


# ============================================
# CORRELATION HEATMAP
# ============================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Air Quality Feature Correlation")

plt.tight_layout()

plt.savefig(
    visualization_folder / "correlation_heatmap.png"
)

plt.show()

plt.close()


print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nVisualizations saved in:")
print(visualization_folder)