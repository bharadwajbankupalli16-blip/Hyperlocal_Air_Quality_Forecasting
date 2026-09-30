import pandas as pd
import joblib

print("=" * 70)
print("CURRENT MULTIMODAL AQI PREDICTION")
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


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("\n1. Loading latest dataset...")

data = pd.read_csv(DATA_FILE)

data["DateTime"] = pd.to_datetime(
    data["DateTime"]
)

print(f"Records available: {len(data)}")


# ---------------------------------------------------------
# Features
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
# Load model
# ---------------------------------------------------------

print("\n2. Loading trained model...")

model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully!")


# ---------------------------------------------------------
# Get latest record for each neighborhood
# ---------------------------------------------------------

print("\n3. Generating predictions...")

latest = (
    data.sort_values("DateTime")
    .groupby("Neighborhood")
    .tail(1)
    .copy()
)


# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------

latest["Predicted_Next_Hour_AQI"] = (
    model.predict(
        latest[features]
    )
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("NEXT-HOUR AQI FORECAST")
print("=" * 70)

for _, row in latest.iterrows():

    print(
        f"\nNeighborhood: "
        f"{row['Neighborhood']}"
    )

    print(
        f"Time: "
        f"{row['DateTime']}"
    )

    print(
        f"Current AQI: "
        f"{row['AQI']:.2f}"
    )

    print(
        f"Predicted next-hour AQI: "
        f"{row['Predicted_Next_Hour_AQI']:.2f}"
    )

    print(
        f"Traffic volume: "
        f"{row['Traffic_Volume']:.2f}"
    )

    print(
        f"Real PM2.5: "
        f"{row['Real_PM2_5']:.2f}"
    )


# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------

output_file = (
    "data/processed/"
    "current_aqi_predictions.csv"
)

latest.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 70)

print("Predictions saved to:")

print(output_file)

print("=" * 70)