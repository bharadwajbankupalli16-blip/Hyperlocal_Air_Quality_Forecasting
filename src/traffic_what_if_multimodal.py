import pandas as pd
import joblib

print("=" * 70)
print("MULTIMODAL TRAFFIC REDUCTION WHAT-IF ANALYSIS")
print("=" * 70)


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

DATA_FILE = (
    "data/processed/"
    "current_ml_features.csv"
)

MODEL_FILE = (
    "models/"
    "current_multimodal_aqi_model.pkl"
)

OUTPUT_FILE = (
    "data/processed/"
    "traffic_scenario_predictions.csv"
)


# ---------------------------------------------------------
# Traffic reduction
# ---------------------------------------------------------

TRAFFIC_REDUCTION = 20


# ---------------------------------------------------------
# Features used by model
# ---------------------------------------------------------

features = [
    "Traffic_Volume",
    "Traffic_Lag_1",
    "Traffic_Lag_3",

    "PM2_5",
    "PM10",
    "NO2",
    "CO",
    "O3",

    "Real_Temperature_C",
    "Real_Humidity_Percent",
    "Real_Wind_Speed_mps",
    "Real_Precipitation_mm",

    "Real_PM2_5",
    "Real_PM10",
    "Real_CO",
    "Real_NO2",
    "Real_O3",
    "Real_SO2",
    "Aerosol_Optical_Depth",

    "Hour",
    "DayOfWeek",
    "IsWeekend",
    "IsRushHour",

    "AQI_Lag_1",
    "AQI_Lag_3",
    "AQI_Lag_6",
    "AQI_Rolling_3H",
    "AQI_Rolling_6H"
]


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print("\n1. Loading data...")

data = pd.read_csv(DATA_FILE)

data["DateTime"] = pd.to_datetime(
    data["DateTime"]
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("\n2. Loading trained model...")

model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully!")


# ---------------------------------------------------------
# Get latest record per neighborhood
# ---------------------------------------------------------

latest = (
    data.sort_values("DateTime")
    .groupby("Neighborhood")
    .tail(1)
    .copy()
)


# ---------------------------------------------------------
# Baseline prediction
# ---------------------------------------------------------

print("\n3. Calculating baseline predictions...")

latest["Baseline_AQI"] = model.predict(
    latest[features]
)


# ---------------------------------------------------------
# Create traffic-reduction scenario
# ---------------------------------------------------------

scenario = latest.copy()

scenario["Traffic_Volume"] = (
    scenario["Traffic_Volume"]
    * (1 - TRAFFIC_REDUCTION / 100)
)

scenario["Traffic_Lag_1"] = (
    scenario["Traffic_Lag_1"]
    * (1 - TRAFFIC_REDUCTION / 100)
)

scenario["Traffic_Lag_3"] = (
    scenario["Traffic_Lag_3"]
    * (1 - TRAFFIC_REDUCTION / 100)
)


# ---------------------------------------------------------
# Scenario prediction
# ---------------------------------------------------------

print("\n4. Calculating traffic-reduction predictions...")

scenario["Scenario_AQI"] = model.predict(
    scenario[features]
)


# ---------------------------------------------------------
# Calculate difference
# ---------------------------------------------------------

scenario["AQI_Change"] = (
    scenario["Baseline_AQI"]
    - scenario["Scenario_AQI"]
)

scenario["Improvement_Percent"] = (
    scenario["AQI_Change"]
    / scenario["Baseline_AQI"]
    * 100
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print(
    f"TRAFFIC REDUCTION SCENARIO: "
    f"{TRAFFIC_REDUCTION}%"
)
print("=" * 70)

for _, row in scenario.iterrows():

    print(
        f"\nNeighborhood: "
        f"{row['Neighborhood']}"
    )

    print(
        f"Baseline predicted AQI: "
        f"{row['Baseline_AQI']:.2f}"
    )

    print(
        f"Scenario predicted AQI: "
        f"{row['Scenario_AQI']:.2f}"
    )

    print(
        f"AQI change: "
        f"{row['AQI_Change']:.2f}"
    )

    print(
        f"Percentage change: "
        f"{row['Improvement_Percent']:.2f}%"
    )


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

output_columns = [
    "DateTime",
    "Neighborhood",
    "Traffic_Volume",
    "Baseline_AQI",
    "Scenario_AQI",
    "AQI_Change",
    "Improvement_Percent"
]

scenario[
    output_columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)

print("Scenario results saved to:")

print(OUTPUT_FILE)

print("=" * 70)