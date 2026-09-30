# 🌍 Hyperlocal Air Quality Forecasting with Multi-Modal Data

A machine-learning based air-quality forecasting system that combines real pollution-monitoring data, historical weather information, and traffic-related features to predict next-hour PM2.5 concentration.

The project also provides an interactive traffic-reduction "what-if" scenario through a Streamlit dashboard.

---

## 🚀 Project Overview

Air pollution can vary significantly between locations and time periods.

This project explores whether combining multiple environmental signals can improve short-term PM2.5 forecasting.

The system combines:

- 🟢 OpenAQ pollution-monitoring data
- 🌦️ Historical weather data from Open-Meteo
- 🚗 Traffic-volume features
- 🤖 Random Forest regression
- 📈 Time-series lag and rolling features
- 🗺️ Monitoring-station map
- 🎚️ Traffic-reduction scenario analysis
- 🖥️ Interactive Streamlit dashboard

---

## 🎯 Objectives

The main objectives are:

1. Collect real-world air-quality measurements.
2. Integrate historical weather information.
3. Engineer temporal and pollution-history features.
4. Forecast next-hour PM2.5.
5. Evaluate the forecasting model.
6. Visualize monitoring stations.
7. Provide an interactive traffic-reduction scenario.

---

## 🏗️ System Architecture

```text
                 ┌──────────────────────┐
                 │       OpenAQ         │
                 │  PM2.5 / PM10 / NO2  │
                 │         / O3         │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Historical Weather   │
                 │    Open-Meteo        │
                 │ Temperature / Wind   │
                 │ Humidity / Rain      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Traffic Features     │
                 │ Prototype Dataset    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Feature Engineering  │
                 │ Lag + Rolling + Time │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Random Forest Model  │
                 │   PM2.5 Forecast     │
                 └──────────┬───────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      Next-Hour Forecast          Traffic What-If
              │                           │
              └─────────────┬─────────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Streamlit Dashboard  │
                 │ Charts + Map + ML    │
                 └──────────────────────┘