import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    saved = joblib.load("fare_model.pkl")
    return saved["model"], saved["features"]

model, feature_cols = load_model()

RATE_CODES = {
    "Standard rate": 1,
    "JFK Airport (flat rate)": 2,
    "Newark Airport": 3,
    "Nassau or Westchester": 4,
    "Negotiated fare": 5,
    "Group ride": 6,
}

# Page layout 
st.set_page_config(page_title="NYC Taxi Fare Predictor")
st.title("NYC Taxi Fare Predictor")
st.write("Enter the trip details below and the model will estimate the meter fare.")

# User inputs 
col1, col2 = st.columns(2)

with col1:
    trip_distance = st.number_input(
        "Trip distance (miles)",
        min_value=0.1, max_value=100.0, value=3.5, step=0.1)

    passenger_count = st.number_input(
        "Number of passengers",
        min_value=1, max_value=6, value=1, step=1)

    rate_label = st.selectbox("Rate type", list(RATE_CODES.keys()))

with col2:
    pickup_date = st.date_input("Pickup date")
    
    st.write("Pickup time")
    t1, t2, t3 = st.columns(3)
    with t1:
        hour_12 = st.selectbox("Hour", list(range(1, 13)), index=11)   
    with t2:
        minute = st.selectbox("Minute", list(range(0, 60)), index=0)  
    with t3:
        am_pm = st.selectbox("AM/PM", ["AM", "PM"])

# Predict taxi fare
if st.button("Predict Fare", type="primary"):

    row = {
        "trip_distance": trip_distance,
        "passenger_count": passenger_count,
        "hour": ((hour_12 % 12) + (12 if am_pm == "PM" else 0)),
        "day_of_week": pickup_date.weekday(),   # Monday = 0 ... Sunday = 6
        "month": pickup_date.month,
    }

    ratecode = RATE_CODES[rate_label]
    row[f"rate_{ratecode}"] = 1

    X_new = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)

    fare = model.predict(X_new)[0]

    st.success(f"### Estimated fare: ${fare:.2f}")
    st.caption("This is the base meter fare. Tips, tolls and surcharges are not included.")

st.divider()
st.caption("Model: Random Forest · trained on 2025 NYC Yellow Taxi data · "
           "average error around $2.63 per trip.")
