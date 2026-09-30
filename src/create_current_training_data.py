import pandas as pd
import numpy as np
import os

print("=" * 70)
print("CREATING CURRENT-PERIOD NEIGHBORHOOD + TRAFFIC DATA")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load real environmental timestamps
# ---------------------------------------------------------

ENV_FILE = "data/processed/real_environment_hyderabad.csv"

environment = pd.read_csv(ENV_FILE)

environment["DateTime"] = pd.to_datetime(
    environment["DateTime"]
)

print(f"\nEnvironmental timestamps: {len(environment)}")


# ---------------------------------------------------------
# 2. Hyderabad neighborhoods
# ---------------------------------------------------------

neighborhoods = [
    "Banjara Hills",
    "Madhapur",
    "Hitech City",
    "Kukatpally",
    "Gachibowli",
    "Secunderabad",
    "Ameerpet",
    "LB Nagar"
]


# ---------------------------------------------------------
# 3. Create one row for each neighborhood/hour
# ---------------------------------------------------------

rows = []

np.random.seed(42)

for _, env_row in environment.iterrows():

    for neighborhood in neighborhoods:

        hour = env_row["DateTime"].hour

        # Base traffic
        traffic = np.random.normal(
            500,
            100
        )

        # Higher traffic during rush hours
        if hour in [8, 9, 17, 18, 19]:

            traffic *= 1.5

        traffic = max(
            50,
            traffic
        )

        # -------------------------------------------------
        # Synthetic prototype AQI
        # -------------------------------------------------

        pm25 = np.random.normal(
            45,
            10
        )

        pm10 = pm25 * np.random.uniform(
            1.5,
            2.0
        )

        no2 = np.random.normal(
            25,
            6
        )

        co = np.random.normal(
            0.8,
            0.2
        )

        o3 = np.random.normal(
            45,
            8
        )

        # Traffic contribution
        traffic_effect = (
            traffic / 500
        ) * 10

        # Simplified prototype AQI
        aqi = (
            pm25 * 1.1
            + pm10 * 0.15
            + no2 * 0.4
            + traffic_effect
        )

        aqi = max(
            10,
            aqi
        )

        rows.append({
            "DateTime": env_row["DateTime"],
            "Neighborhood": neighborhood,
            "Traffic_Volume": round(
                traffic,
                2
            ),
            "PM2_5": round(
                pm25,
                2
            ),
            "PM10": round(
                pm10,
                2
            ),
            "NO2": round(
                no2,
                2
            ),
            "CO": round(
                co,
                2
            ),
            "O3": round(
                o3,
                2
            ),
            "AQI": round(
                aqi,
                2
            )
        })


# ---------------------------------------------------------
# 4. Create DataFrame
# ---------------------------------------------------------

data = pd.DataFrame(rows)


# ---------------------------------------------------------
# 5. Sort
# ---------------------------------------------------------

data = data.sort_values(
    [
        "Neighborhood",
        "DateTime"
    ]
).reset_index(drop=True)


# ---------------------------------------------------------
# 6. Save
# ---------------------------------------------------------

output_file = (
    "data/processed/"
    "current_neighborhood_training_data.csv"
)

os.makedirs(
    "data/processed",
    exist_ok=True
)

data.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# 7. Display
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CURRENT TRAINING DATA CREATED")
print("=" * 70)

print(f"\nRecords: {len(data)}")

print(
    f"Neighborhoods: "
    f"{data['Neighborhood'].nunique()}"
)

print(
    f"Start: {data['DateTime'].min()}"
)

print(
    f"End: {data['DateTime'].max()}"
)

print("\nSample:")

print(
    data.head(10).to_string(
        index=False
    )
)

print("\nSaved to:")

print(output_file)

print("=" * 70)