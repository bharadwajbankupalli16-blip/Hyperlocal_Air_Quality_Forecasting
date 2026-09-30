
import os
import requests
import pandas as pd
from dotenv import load_dotenv


# ============================================================
# OPENAQ HYDERABAD MONITORING STATIONS
# ============================================================

print("=" * 70)
print("OPENAQ HYDERABAD MONITORING STATIONS")
print("=" * 70)


# Load API key
load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    print("\nERROR: OPENAQ_API_KEY was not found in .env")
    raise SystemExit(1)


# OpenAQ endpoint
URL = "https://api.openaq.org/v3/locations"


# Hyderabad bounding box
# xmin, ymin, xmax, ymax
params = {
    "bbox": "78.25,17.25,78.65,17.55",
    "limit": 100,
    "page": 1
}


headers = {
    "X-API-Key": API_KEY
}


try:

    response = requests.get(
        URL,
        params=params,
        headers=headers,
        timeout=30
    )

    print(f"\nHTTP Status: {response.status_code}")

    response.raise_for_status()

    result = response.json()

    locations = result.get("results", [])

    print(f"Monitoring locations found: {len(locations)}")


    records = []


    for location in locations:

        coordinates = location.get(
            "coordinates",
            {}
        )

        records.append(
            {
                "Location_ID": location.get("id"),
                "Station_Name": location.get("name"),
                "Locality": location.get("locality"),
                "Country": location.get("country", {}).get("name"),
                "Latitude": coordinates.get("latitude"),
                "Longitude": coordinates.get("longitude"),
                "Is_Monitor": location.get("isMonitor"),
                "Is_Mobile": location.get("isMobile"),
                "Timezone": location.get("timezone"),
                "Provider": location.get("provider", {}).get("name"),
                "Owner": location.get("owner", {}).get("name")
            }
        )


    data = pd.DataFrame(records)


    if data.empty:

        print("\nNo monitoring stations were returned.")

    else:

        os.makedirs(
            "data/raw",
            exist_ok=True
        )

        output_file = (
            "data/raw/"
            "openaq_hyderabad_stations.csv"
        )

        data.to_csv(
            output_file,
            index=False
        )


        print("\n" + "=" * 70)
        print("SUCCESS")
        print("=" * 70)

        print(
            f"\nSaved {len(data)} monitoring stations."
        )

        print(
            f"\nFile:\n{output_file}"
        )


        print("\nStation information:")

        print(
            data[
                [
                    "Location_ID",
                    "Station_Name",
                    "Latitude",
                    "Longitude",
                    "Provider"
                ]
            ].to_string(index=False)
        )


except requests.exceptions.HTTPError as e:

    print("\nOpenAQ API request failed.")

    print(e)

    if response.status_code == 401:

        print(
            "\nAuthentication failed."
            "\nCheck or regenerate your OpenAQ API key."
        )

    elif response.status_code == 403:

        print(
            "\nAccess was forbidden by OpenAQ."
        )


except requests.exceptions.RequestException as e:

    print("\nNetwork/API request failed.")

    print(e)


print("\n" + "=" * 70)
print("OPENAQ STATION DOWNLOAD FINISHED")
print("=" * 70)