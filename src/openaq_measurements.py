import os
import requests
import pandas as pd
from dotenv import load_dotenv


print("=" * 70)
print("DOWNLOADING OPENAQ POLLUTANT MEASUREMENTS")
print("=" * 70)


# ------------------------------------------------------------
# LOAD API KEY
# ------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    print("ERROR: OPENAQ_API_KEY not found.")
    raise SystemExit(1)


# ------------------------------------------------------------
# LOAD STATIONS
# ------------------------------------------------------------

STATION_FILE = "data/raw/openaq_hyderabad_stations.csv"

if not os.path.exists(STATION_FILE):

    print(
        f"\nERROR: {STATION_FILE} does not exist."
    )

    print(
        "\nRun this first:"
    )

    print(
        "python src\\openaq_api.py"
    )

    raise SystemExit(1)


stations = pd.read_csv(STATION_FILE)

print(
    f"\nStations available: {len(stations)}"
)


# ------------------------------------------------------------
# API SETTINGS
# ------------------------------------------------------------

BASE_URL = "https://api.openaq.org/v3"

HEADERS = {
    "X-API-Key": API_KEY
}


# ------------------------------------------------------------
# DOWNLOAD SENSOR INFORMATION
# ------------------------------------------------------------

sensor_records = []


for _, station in stations.iterrows():

    location_id = int(station["Location_ID"])

    url = (
        f"{BASE_URL}/locations/"
        f"{location_id}/sensors"
    )

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        if response.status_code != 200:

            print(
                f"Location {location_id}: "
                f"HTTP {response.status_code}"
            )

            continue


        result = response.json()

        sensors = result.get(
            "results",
            []
        )


        for sensor in sensors:

            parameter = sensor.get(
                "parameter",
                {}
            )

            sensor_records.append(
                {
                    "Location_ID": location_id,
                    "Station_Name": station[
                        "Station_Name"
                    ],
                    "Sensor_ID": sensor.get(
                        "id"
                    ),
                    "Parameter": parameter.get(
                        "name"
                    ),
                    "Units": parameter.get(
                        "units"
                    ),
                    "Display_Name": parameter.get(
                        "displayName"
                    ),
                    "Latitude": station[
                        "Latitude"
                    ],
                    "Longitude": station[
                        "Longitude"
                    ]
                }
            )


    except requests.exceptions.RequestException as e:

        print(
            f"Location {location_id} failed: {e}"
        )


# ------------------------------------------------------------
# SAVE SENSOR INFORMATION
# ------------------------------------------------------------

sensor_data = pd.DataFrame(
    sensor_records
)


if sensor_data.empty:

    print(
        "\nNo sensor information was returned."
    )

else:

    os.makedirs(
        "data/raw",
        exist_ok=True
    )

    output_file = (
        "data/raw/"
        "openaq_hyderabad_sensors.csv"
    )

    sensor_data.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)

    print(
        f"Sensors found: {len(sensor_data)}"
    )

    print(
        f"\nSaved:\n{output_file}"
    )

    print("\nPollutant types:")

    print(
        sensor_data[
            "Parameter"
        ].value_counts().to_string()
    )

    print("\nFirst 20 sensors:")

    print(
        sensor_data.head(20).to_string(
            index=False
        )
    )


print("\n" + "=" * 70)
print("SENSOR DISCOVERY COMPLETED")
print("=" * 70)