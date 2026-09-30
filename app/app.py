import streamlit as st
import pandas as pd
import joblib
import folium

from pathlib import Path
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hyperlocal Air Quality Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_FILE = "data/processed/real_ml_features.csv"

MODEL_FILE = "models/real_pm25_random_forest.pkl"

METRICS_FILE = (
    "data/processed/real_pm25_model_metrics.csv"
)

PREDICTIONS_FILE = (
    "data/processed/real_pm25_model_predictions.csv"
)

IMPORTANCE_FILE = (
    "data/processed/real_pm25_feature_importance.csv"
)


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [
    "OpenAQ_PM2_5",
    "OpenAQ_PM10",
    "OpenAQ_NO2",
    "OpenAQ_O3",

    "Real_Temperature_C",
    "Real_Humidity_Percent",
    "Real_Wind_Speed_mps",
    "Real_Precipitation_mm",

    "Traffic_Volume",
    "Traffic_Lag_1",
    "Traffic_Lag_3",

    "PM2_5_Lag_1",
    "PM2_5_Lag_3",
    "PM2_5_Lag_6",
    "PM2_5_Lag_24",

    "PM2_5_Rolling_3H",
    "PM2_5_Rolling_6H",

    "PM10_Lag_1",
    "NO2_Lag_1",
    "O3_Lag_1",

    "Temperature_Lag_1",
    "Humidity_Lag_1",
    "Wind_Lag_1",

    "Hour",
    "DayOfWeek",
    "Month",
    "IsWeekend",
    "IsRushHour"
]


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not Path(DATA_FILE).exists():
        return pd.DataFrame()

    data = pd.read_csv(DATA_FILE)

    data["DateTime"] = pd.to_datetime(
        data["DateTime"],
        utc=True
    )

    return data


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not Path(MODEL_FILE).exists():
        return None

    return joblib.load(MODEL_FILE)


# ============================================================
# LOAD METRICS
# ============================================================

@st.cache_data
def load_metrics():

    if not Path(METRICS_FILE).exists():
        return pd.DataFrame()

    return pd.read_csv(METRICS_FILE)


# ============================================================
# LOAD TEST PREDICTIONS
# ============================================================

@st.cache_data
def load_predictions():

    if not Path(PREDICTIONS_FILE).exists():
        return pd.DataFrame()

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    predictions["DateTime"] = pd.to_datetime(
        predictions["DateTime"],
        utc=True
    )

    return predictions


# ============================================================
# LOAD FEATURE IMPORTANCE
# ============================================================

@st.cache_data
def load_importance():

    if not Path(IMPORTANCE_FILE).exists():
        return pd.DataFrame()

    return pd.read_csv(
        IMPORTANCE_FILE
    )


# ============================================================
# INITIALIZE
# ============================================================

data = load_data()

model = load_model()

metrics = load_metrics()

predictions = load_predictions()

importance = load_importance()


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if data.empty:

    st.error(
        "The real ML dataset was not found.\n\n"
        "Please run:\n"
        "`python src\\prepare_real_features.py`"
    )

    st.stop()


if model is None:

    st.error(
        "The trained model was not found.\n\n"
        "Please run:\n"
        "`python src\\train_real_pm25_model.py`"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌍 Hyperlocal Air Quality Forecasting"
)

st.markdown(
    """
### Multimodal PM2.5 Forecasting Dashboard

This project combines real air-quality monitoring data,
historical weather information, and a prototype traffic
feature to forecast next-hour PM2.5 concentration.
"""
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

info1, info2, info3, info4 = st.columns(4)

info1.metric(
    "Monitoring Stations",
    data["Station_Name"].nunique()
)

info2.metric(
    "Dataset Records",
    len(data)
)

info3.metric(
    "ML Features",
    len(FEATURES)
)

info4.metric(
    "Forecast Horizon",
    "Next Hour"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Dashboard Controls")

stations = sorted(
    data["Station_Name"]
    .dropna()
    .unique()
)

selected_station = st.sidebar.selectbox(
    "📍 Monitoring Station",
    stations
)

traffic_reduction = st.sidebar.slider(
    "🚗 Traffic Reduction Scenario (%)",
    min_value=0,
    max_value=50,
    value=20,
    step=5
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
**Data sources**

Pollution: OpenAQ

Weather: Open-Meteo Historical Archive

Traffic: Synthetic prototype

Model: Random Forest
"""
)


# ============================================================
# SELECT STATION
# ============================================================

station_data = data[
    data["Station_Name"] == selected_station
].copy()

station_data = station_data.sort_values(
    "DateTime"
)

if station_data.empty:

    st.warning(
        "No data available for this monitoring station."
    )

    st.stop()


# ============================================================
# LATEST RECORD
# ============================================================

latest = station_data.tail(1).copy()


# ============================================================
# BASELINE PREDICTION
# ============================================================

try:

    baseline_prediction = model.predict(
        latest[FEATURES]
    )[0]

except Exception as e:

    st.error(
        f"Prediction failed: {e}"
    )

    st.stop()


# ============================================================
# TRAFFIC WHAT-IF SCENARIO
# ============================================================

scenario_input = latest[FEATURES].copy()

reduction_factor = (
    1 - traffic_reduction / 100
)

scenario_input["Traffic_Volume"] *= (
    reduction_factor
)

scenario_input["Traffic_Lag_1"] *= (
    reduction_factor
)

scenario_input["Traffic_Lag_3"] *= (
    reduction_factor
)


try:

    scenario_prediction = model.predict(
        scenario_input
    )[0]

except Exception as e:

    st.error(
        f"Scenario prediction failed: {e}"
    )

    st.stop()


prediction_change = (
    scenario_prediction
    - baseline_prediction
)


if baseline_prediction != 0:

    reduction_percent = (
        (
            baseline_prediction
            - scenario_prediction
        )
        / baseline_prediction
        * 100
    )

else:

    reduction_percent = 0


# ============================================================
# PM2.5 CATEGORY
# ============================================================

def pm25_category(value):

    if value <= 12:

        return "Good"

    elif value <= 35.4:

        return "Moderate"

    elif value <= 55.4:

        return "Unhealthy for Sensitive Groups"

    elif value <= 150.4:

        return "Unhealthy"

    elif value <= 250.4:

        return "Very Unhealthy"

    else:

        return "Hazardous"


# ============================================================
# STATION HEADER
# ============================================================

st.header(
    f"📍 {selected_station}"
)

st.caption(
    f"Latest available record: "
    f"{latest['DateTime'].iloc[0]}"
)


# ============================================================
# MAIN METRICS
# ============================================================

st.subheader(
    "📊 Air Quality Forecast"
)

col1, col2, col3, col4 = st.columns(4)


current_pm25 = latest[
    "OpenAQ_PM2_5"
].iloc[0]

col1.metric(
    "Current PM2.5",
    f"{current_pm25:.2f} µg/m³"
)

col2.metric(
    "Next-Hour Forecast",
    f"{baseline_prediction:.2f} µg/m³"
)

col3.metric(
    f"{traffic_reduction}% Traffic Scenario",
    f"{scenario_prediction:.2f} µg/m³"
)

col4.metric(
    "Scenario Change",
    f"{prediction_change:+.2f} µg/m³"
)


st.info(
    f"Forecast PM2.5 category: "
    f"**{pm25_category(baseline_prediction)}**"
)


# ============================================================
# CURRENT ENVIRONMENT
# ============================================================

st.subheader(
    "🌦️ Environmental Conditions"
)

c1, c2, c3, c4, c5 = st.columns(5)


temperature = latest[
    "Real_Temperature_C"
].iloc[0]

humidity = latest[
    "Real_Humidity_Percent"
].iloc[0]

wind = latest[
    "Real_Wind_Speed_mps"
].iloc[0]

pm10 = latest[
    "OpenAQ_PM10"
].iloc[0]

traffic = latest[
    "Traffic_Volume"
].iloc[0]


c1.metric(
    "Temperature",
    f"{temperature:.1f} °C"
)

c2.metric(
    "Humidity",
    f"{humidity:.1f}%"
)

c3.metric(
    "Wind Speed",
    f"{wind:.1f} m/s"
)

c4.metric(
    "PM10",
    f"{pm10:.1f} µg/m³"
)

c5.metric(
    "Traffic",
    f"{traffic:.0f}"
)


# ============================================================
# PM2.5 HISTORY
# ============================================================

st.subheader(
    "📈 PM2.5 History"
)

pm25_history = station_data[
    [
        "DateTime",
        "OpenAQ_PM2_5"
    ]
].copy()

pm25_history = pm25_history.set_index(
    "DateTime"
)

st.line_chart(
    pm25_history["OpenAQ_PM2_5"],
    use_container_width=True
)


# ============================================================
# POLLUTANT HISTORY
# ============================================================

st.subheader(
    "🧪 Pollutant History"
)

pollution_history = station_data[
    [
        "DateTime",
        "OpenAQ_PM2_5",
        "OpenAQ_PM10",
        "OpenAQ_NO2",
        "OpenAQ_O3"
    ]
].copy()

pollution_history = pollution_history.set_index(
    "DateTime"
)

st.line_chart(
    pollution_history,
    use_container_width=True
)


# ============================================================
# WEATHER HISTORY
# ============================================================

st.subheader(
    "🌦️ Weather History"
)

weather_history = station_data[
    [
        "DateTime",
        "Real_Temperature_C",
        "Real_Humidity_Percent",
        "Real_Wind_Speed_mps"
    ]
].copy()

weather_history = weather_history.set_index(
    "DateTime"
)

st.line_chart(
    weather_history,
    use_container_width=True
)


# ============================================================
# TRAFFIC HISTORY
# ============================================================

st.subheader(
    "🚗 Traffic History"
)

traffic_history = station_data[
    [
        "DateTime",
        "Traffic_Volume"
    ]
].copy()

traffic_history = traffic_history.set_index(
    "DateTime"
)

st.line_chart(
    traffic_history,
    use_container_width=True
)


# ============================================================
# TRAFFIC WHAT-IF
# ============================================================

st.subheader(
    "🚦 Traffic Reduction What-If Analysis"
)

st.write(
    f"""
The slider currently simulates a **{traffic_reduction}%**
reduction in traffic volume and the corresponding traffic
lag features.
"""
)

scenario_data = pd.DataFrame(
    {
        "Scenario": [
            "Baseline",
            f"Traffic - {traffic_reduction}%"
        ],
        "Predicted PM2.5": [
            baseline_prediction,
            scenario_prediction
        ]
    }
)

st.bar_chart(
    scenario_data.set_index(
        "Scenario"
    ),
    use_container_width=True
)


if prediction_change < 0:

    st.success(
        f"Under this model scenario, predicted PM2.5 "
        f"changes by {prediction_change:.2f} µg/m³ "
        f"({abs(reduction_percent):.2f}% lower)."
    )

elif prediction_change > 0:

    st.warning(
        f"Under this model scenario, predicted PM2.5 "
        f"changes by +{prediction_change:.2f} µg/m³."
    )

else:

    st.info(
        "The model predicts no change for this scenario."
    )


st.caption(
    "This is a model-based what-if simulation. "
    "It is not a causal estimate of the real-world "
    "effect of traffic reduction."
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader(
    "🤖 Model Performance"
)

if not metrics.empty:

    try:

        mae = metrics.loc[
            metrics["Metric"] == "MAE",
            "Value"
        ].iloc[0]

        rmse = metrics.loc[
            metrics["Metric"] == "RMSE",
            "Value"
        ].iloc[0]

        r2 = metrics.loc[
            metrics["Metric"] == "R2",
            "Value"
        ].iloc[0]

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "MAE",
            f"{mae:.3f} µg/m³"
        )

        m2.metric(
            "RMSE",
            f"{rmse:.3f} µg/m³"
        )

        m3.metric(
            "R²",
            f"{r2:.3f}"
        )

    except Exception as e:

        st.warning(
            f"Could not read model metrics: {e}"
        )

else:

    st.warning(
        "Model metrics file not found. "
        "Run evaluate_real_pm25_model.py."
    )


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

st.subheader(
    "🎯 Actual vs Predicted PM2.5"
)

if not predictions.empty:

    actual_predicted = predictions[
        [
            "DateTime",
            "Actual_PM2_5",
            "Predicted_PM2_5"
        ]
    ].copy()

    actual_predicted = (
        actual_predicted
        .set_index("DateTime")
    )

    st.line_chart(
        actual_predicted,
        use_container_width=True
    )

    st.caption(
        "Actual values are measured OpenAQ PM2.5 "
        "values from the chronological test period. "
        "Predicted values are generated by the Random "
        "Forest model."
    )

else:

    st.warning(
        "Prediction evaluation data not found."
    )


# ============================================================
# PREDICTION ERROR
# ============================================================

st.subheader(
    "📉 Prediction Error"
)

if not predictions.empty:

    error_history = predictions[
        [
            "DateTime",
            "Prediction_Error"
        ]
    ].copy()

    error_history = (
        error_history
        .set_index("DateTime")
    )

    st.line_chart(
        error_history["Prediction_Error"],
        use_container_width=True
    )

    error_mean = predictions[
        "Prediction_Error"
    ].mean()

    error_std = predictions[
        "Prediction_Error"
    ].std()

    e1, e2 = st.columns(2)

    e1.metric(
        "Mean Error",
        f"{error_mean:.3f} µg/m³"
    )

    e2.metric(
        "Error Std. Dev.",
        f"{error_std:.3f} µg/m³"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "🔍 Feature Importance"
)

if not importance.empty:

    top_features = (
        importance
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(15)
    )

    st.bar_chart(
        top_features.set_index(
            "Feature"
        )["Importance"],
        use_container_width=True
    )

    with st.expander(
        "View complete feature importance table"
    ):

        st.dataframe(
            importance,
            use_container_width=True
        )

else:

    st.warning(
        "Feature importance file not found."
    )


# ============================================================
# MONITORING STATION MAP
# ============================================================

st.subheader(
    "🗺️ OpenAQ Monitoring Stations"
)

# Only use records that actually contain coordinates
coordinates = data[
    [
        "Station_Name",
        "Latitude",
        "Longitude"
    ]
].copy()

coordinates = coordinates.dropna(
    subset=[
        "Latitude",
        "Longitude"
    ]
)

coordinates = coordinates.drop_duplicates(
    subset=["Station_Name"]
)


if not coordinates.empty:

    station_summary = (
        data.groupby("Station_Name")
        .agg(
            Average_PM2_5=(
                "OpenAQ_PM2_5",
                "mean"
            )
        )
        .reset_index()
    )

    station_summary = station_summary.merge(
        coordinates,
        on="Station_Name",
        how="inner"
    )

    map_center = [
        station_summary["Latitude"].mean(),
        station_summary["Longitude"].mean()
    ]

    m = folium.Map(
        location=map_center,
        zoom_start=11
    )

    for _, row in station_summary.iterrows():

        popup = folium.Popup(
            f"""
            <b>{row['Station_Name']}</b><br>
            Average PM2.5:
            {row['Average_PM2_5']:.2f} µg/m³
            """,
            max_width=300
        )

        folium.CircleMarker(
            location=[
                row["Latitude"],
                row["Longitude"]
            ],
            radius=8,
            popup=popup,
            tooltip=row["Station_Name"],
            fill=True
        ).add_to(m)

    st_folium(
        m,
        width=1100,
        height=500
    )

else:

    st.info(
        "No valid station coordinates are available "
        "in the current OpenAQ dataset."
    )


# ============================================================
# STATION COMPARISON
# ============================================================

st.subheader(
    "📊 Monitoring Station Comparison"
)

station_comparison = (
    data.groupby("Station_Name")
    .agg(
        Average_PM2_5=(
            "OpenAQ_PM2_5",
            "mean"
        ),
        Maximum_PM2_5=(
            "OpenAQ_PM2_5",
            "max"
        ),
        Records=(
            "OpenAQ_PM2_5",
            "count"
        )
    )
    .reset_index()
    .sort_values(
        "Average_PM2_5",
        ascending=False
    )
)

st.dataframe(
    station_comparison,
    use_container_width=True
)


# ============================================================
# LATEST MONITORING DATA
# ============================================================

st.subheader(
    "📋 Latest Monitoring Records"
)

latest_columns = [
    "Station_Name",
    "DateTime",
    "OpenAQ_PM2_5",
    "OpenAQ_PM10",
    "OpenAQ_NO2",
    "OpenAQ_O3",
    "Real_Temperature_C",
    "Real_Humidity_Percent",
    "Real_Wind_Speed_mps",
    "Traffic_Volume"
]

available_columns = [
    column
    for column in latest_columns
    if column in data.columns
]

latest_table = (
    data
    .sort_values("DateTime")
    .tail(25)
)

st.dataframe(
    latest_table[available_columns],
    use_container_width=True
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.subheader(
    "📚 Dataset Information"
)

d1, d2 = st.columns(2)

with d1:

    st.markdown(
        """
### Pollution

**Source:** OpenAQ

Measured pollutants include:

- PM2.5
- PM10
- NO₂
- O₃
"""
    )

with d2:

    st.markdown(
        """
### Weather

**Source:** Open-Meteo Historical Archive

Variables include:

- Temperature
- Humidity
- Wind speed
- Precipitation
"""
    )


# ============================================================
# METHODOLOGY
# ============================================================

st.subheader(
    "🧠 Methodology"
)

st.markdown(
    """
1. Collect historical OpenAQ pollution measurements.
2. Download matching historical weather observations.
3. Create prototype traffic-volume features.
4. Generate time, lag, and rolling-window features.
5. Predict next-hour PM2.5.
6. Use an 80/20 chronological train-test split.
7. Train a Random Forest regression model.
8. Evaluate using MAE, RMSE and R².
9. Simulate traffic-reduction scenarios.
10. Visualize monitoring stations and model results.
"""
)


# ============================================================
# LIMITATIONS
# ============================================================

st.subheader(
    "⚠️ Limitations"
)

st.markdown(
    """
- Traffic data is currently a **synthetic prototype**, not
  verified historical traffic-sensor data.
- OpenAQ station coverage varies by location and time.
- The model predicts PM2.5 concentration, not an official
  regulatory AQI.
- The traffic slider represents a model-based scenario and
  does not establish a causal relationship.
- The dashboard is intended for research and educational
  demonstration, not official air-quality monitoring or
  health advice.
"""
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Hyperlocal Air Quality Forecasting | "
    "Python • OpenAQ • Open-Meteo • Scikit-learn • "
    "Random Forest • Streamlit • Folium"
)