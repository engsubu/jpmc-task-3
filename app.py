import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="JPMC Task 3 - Predictive Analytics", layout="wide")
st.title("📈 Predictive Analytics - JPMC Task 3")
st.write("Forecast future trends using Facebook Prophet")

DATA_URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df.columns = ['ds', 'y']  # Prophet requires this
    df['ds'] = pd.to_datetime(df['ds'])
    return df

df = load_data()

st.write("### 1. Historical Data")
st.dataframe(df.tail(10), use_container_width=True)
st.write(f"**Records:** {len(df)} | **From:** {df['ds'].min().date()} to {df['ds'].max().date()}")

# Train Prophet model
with st.spinner("Training Prophet model..."):
    m = Prophet(yearly_seasonality=True)
    m.fit(df)

# CRITICAL FIX: Use 'MS' = Month Start. 'M' will crash on new pandas
future = m.make_future_dataframe(periods=12, freq='MS') 
forecast = m.predict(future)

st.write("### 2. Forecast Next 12 Months")
fig1 = m.plot(forecast)
plt.xlabel("Date")
plt.ylabel("Passengers")
st.pyplot(fig1)

st.write("### 3. Trend + Seasonality")
fig2 = m.plot_components(forecast)
st.pyplot(fig2)

# Evaluation on last 12 months
st.write("### 4. Model Accuracy")
test_df = df.tail(12).reset_index(drop=True)
pred_df = forecast[['ds', 'yhat']].tail(12).reset_index(drop=True)

mae = abs(test_df['y'] - pred_df['yhat']).mean()
rmse = ((test_df['y'] - pred_df['yhat'])**2).mean()**0.5

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

st.write("### 5. Forecast Table")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12), use_container_width=True)
