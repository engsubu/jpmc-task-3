import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("Predictive Analytics - JPMC Task 3")
st.write("Forecast future trends using Prophet")

# Use a sample dataset - Air Passengers
DATA_URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df.columns = ['ds', 'y'] # Prophet needs columns named ds=date, y=value
    df['ds'] = pd.to_datetime(df['ds'])
    return df

df = load_data()
st.write("### Historical Data")
st.dataframe(df.tail())

# Train model
m = Prophet()
m.fit(df)

# Make future dataframe for 12 months
future = m.make_future_dataframe(periods=12, freq='M')
forecast = m.predict(future)

# Plot
st.write("### Forecast for Next 12 Months")
fig1 = m.plot(forecast)
st.pyplot(fig1)

fig2 = m.plot_components(forecast)
st.pyplot(fig2)

# Accuracy - last 12 months as test
st.write("### Model Accuracy on Test Data")
test_df = df.tail(12)
pred_df = forecast[['ds','yhat']].tail(12)
mae = abs(test_df['y'].values - pred_df['yhat'].values).mean()
st.metric("MAE - Mean Absolute Error", round(mae,2))
