import os
import requests
import pandas as pd
from dotenv import load_dotenv


print("=" * 70)
print("DOWNLOADING REAL OPENAQ MEASUREMENTS")
print("=" * 70)


# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    print("ERROR: OPENAQ_API_KEY not found.")
    raise SystemExit(1)


# ============================================================
# LOAD SENSORS
# ============================================================

SENSOR_FILE = "data/raw/openaq_hyderabad_sensors.csv"

if not os.path.exists(SENSOR_FILE):
    print(f"ERROR: {SENSOR_FILE} not found.")
    raise SystemExit(1)


sensors = pd.read_csv(SENSOR_FILE)

print(f"\nTotal sensors: {len(sensors)}")


# ============================================================
# SELECT POLLUTANTS
# ============================================================

wanted = {
    "pm25",
    "pm10",
    "no2",
    "o3"
}

sensors["Parameter"] = (
    sensors["Parameter"]
    .astype(str)
    .str.strip()
    .str.lower()
)

sensors = sensors[
    sensors["Parameter"].isin(wanted)
].drop_duplicates(
    subset=["Sensor_ID"]
)


# Use up to 20 sensors for the prototype

sensors = sensors.head(20)

print(
    f"Pollution sensors selected: {len(sensors)}"
)


# ============================================================
# API
# ============================================================

BASE_URL = "https://api.openaq.org/v3"

HEADERS = {
    "X-API-Key": API_KEY
}


records = []


# ============================================================
# DOWNLOAD MEASUREMENTS
# ============================================================

for index, (_, sensor) in enumerate(
    sensors.iterrows(),
    start=1
):

    sensor_id = int(
        sensor["Sensor_ID"]
    )

    parameter = sensor["Parameter"]

    print(
        f"[{index}/{len(sensors)}] "
        f"{parameter} sensor {sensor_id}"
    )


    url = (
        f"{BASE_URL}/sensors/"
        f"{sensor_id}/measurements"
    )


    params = {
        "limit": 100,
        "page": 1
    }


    try:

        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=20
        )


        if response.status_code != 200:

            print(
                f"   HTTP {response.status_code}"
            )

            continue


        result = response.json()

        measurements = result.get(
            "results",
            []
        )


        print(
            f"   Measurements: "
            f"{len(measurements)}"
        )


        for measurement in measurements:

            # =================================================
            # OPENAQ V3 TIMESTAMP
            # =================================================

            period = measurement.get(
                "period",
                {}
            )

            datetime_from = period.get(
                "datetimeFrom",
                {}
            )

            datetime_to = period.get(
                "datetimeTo",
                {}
            )


            timestamp_from = datetime_from.get(
                "utc"
            )

            timestamp_to = datetime_to.get(
                "utc"
            )


            # =================================================
            # MEASUREMENT VALUE
            # =================================================

            value = measurement.get(
                "value"
            )


            records.append(
                {
                    "Location_ID":
                        sensor["Location_ID"],

                    "Station_Name":
                        sensor["Station_Name"],

                    "Sensor_ID":
                        sensor_id,

                    "Parameter":
                        parameter,

                    "Value":
                        value,

                    "Unit":
                        sensor.get(
                            "Units"
                        ),

                    "DateTime_From":
                        timestamp_from,

                    "DateTime_To":
                        timestamp_to,

                    "Latitude":
                        sensor["Latitude"],

                    "Longitude":
                        sensor["Longitude"]
                }
            )


    except requests.exceptions.RequestException as e:

        print(
            f"   Request failed: {e}"
        )


# ============================================================
# DATAFRAME
# ============================================================

data = pd.DataFrame(records)


if data.empty:

    print(
        "\nERROR: No measurements downloaded."
    )

    raise SystemExit(1)


print(
    f"\nDownloaded records: {len(data)}"
)


# ============================================================
# CONVERT TIMESTAMPS
# ============================================================

data["DateTime_From"] = pd.to_datetime(
    data["DateTime_From"],
    errors="coerce",
    utc=True
)

data["DateTime_To"] = pd.to_datetime(
    data["DateTime_To"],
    errors="coerce",
    utc=True
)


# ============================================================
# CONVERT VALUES
# ============================================================

data["Value"] = pd.to_numeric(
    data["Value"],
    errors="coerce"
)


# ============================================================
# REMOVE INVALID
# ============================================================

data = data.dropna(
    subset=[
        "DateTime_From",
        "Value"
    ]
).copy()


data = data[
    data["Value"] >= 0
]


# ============================================================
# SAVE
# ============================================================

output_file = (
    "data/raw/"
    "openaq_hyderabad_measurements.csv"
)


data.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    f"\nValid measurements: {len(data)}"
)

print(
    "\nValid timestamps:",
    data["DateTime_From"].notna().sum()
)


print("\nPollutants:")

print(
    data["Parameter"]
    .value_counts()
    .to_string()
)


print("\nTime range:")

print(
    "Start:",
    data["DateTime_From"].min()
)

print(
    "End:",
    data["DateTime_From"].max()
)


print("\nSample:")

print(
    data.head(5).to_string(
        index=False
    )
)


print(
    f"\nSaved:\n{output_file}"
)


print("\n" + "=" * 70)
print("OPENAQ DOWNLOAD COMPLETED")
print("=" * 70)