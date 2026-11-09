import pickle
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

# ---- Settings ----------------------------------------------------------
CITIES = {
    "Pittsburgh": (40.44, -79.99),
    "New York": (40.71, -74.01),
    "Chicago": (41.88, -87.63),
    "Miami": (25.76, -80.19),
    "Los Angeles": (34.05, -118.24),
}
FEATURES = ["temperature", "is_weekend", "rain_chance"]   # same names and order as in training
LOW_TEMP, HIGH_TEMP = 50, 100        # the temperatures (F) the model learned from
LOW_SALES, HIGH_SALES = 200, 300     # below LOW_SALES is a slow day, above HIGH_SALES is a busy one
URL = "https://api.open-meteo.com/v1/forecast"
# ------------------------------------------------------------------------

# Load the trained model (we trained it once, in Colab, and saved it to a file)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Ice Cream Sales Predictor", page_icon="🍦")
st.title("🍦 Ice Cream Sales Predictor")
st.caption("Pick a city and a date. The app gets the forecast (high temperature and chance of rain), "
           "checks if it is a weekend, and predicts how many cones you will sell.")

city = st.selectbox("City", list(CITIES))
selected_date = st.date_input(
    "Date",
    value=date.today() + timedelta(days=1),
    min_value=date.today(),
    max_value=date.today() + timedelta(days=15),    # the forecast reaches about 16 days ahead
)

if st.button("Predict sales", type="primary"):
    # 1. Ask the weather API for the forecast high and the chance of rain
    latitude, longitude = CITIES[city]
    response = requests.get(
        URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,precipitation_probability_max",
            "temperature_unit": "fahrenheit",
            "timezone": "auto",
            "start_date": str(selected_date),
            "end_date": str(selected_date),
        },
        timeout=30,
    )
    response.raise_for_status()
    daily = response.json()["daily"]
    temperature = daily["temperature_2m_max"][0]
    rain_chance = daily["precipitation_probability_max"][0]
    if rain_chance is None:              # far-off days sometimes have no rain forecast yet
        rain_chance = 0
        st.caption("No rain forecast for this day yet, so the app assumes 0% chance of rain.")

    # 2. Weekend or weekday? Saturday is 5 and Sunday is 6
    is_weekend = 1 if selected_date.weekday() >= 5 else 0

    # 3. Give the model the same inputs it was trained on, in the same order, then predict
    features = pd.DataFrame([[temperature, is_weekend, rain_chance]], columns=FEATURES)
    prediction = max(0, model.predict(features)[0])

    # 4. Show the result
    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast high", f"{temperature:.0f} °F")
    col2.metric("Chance of rain", f"{rain_chance:.0f}%")
    col3.metric("Day", "Weekend" if is_weekend else "Weekday")
    st.metric("Predicted sales", f"{prediction:.0f} cones")

    if temperature < LOW_TEMP or temperature > HIGH_TEMP:
        st.warning(f"This temperature is outside the range the model learned from ({LOW_TEMP} to {HIGH_TEMP} °F). "
                   "Treat the number with caution.")

    if prediction > HIGH_SALES:
        st.info("Busy day: make sure you are fully stocked.")
    elif prediction > LOW_SALES:
        st.info("A normal day is expected.")
    else:
        st.info("A slow day: consider a promotion, or order less.")

    with st.expander("How the model decides"):
        st.write(
            f"Each extra degree adds about **{model.coef_[0]:.1f} cones**. "
            f"A weekend adds about **{model.coef_[1]:.0f} cones**. "
            f"Each extra 10% chance of rain changes sales by about **{model.coef_[2] * 10:.0f} cones**."
        )
