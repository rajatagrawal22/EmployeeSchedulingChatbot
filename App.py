import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
# from SQL_Data_Import import df

# Load trained model and scaler
model = load_model('lstm_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load data for recent input
data = pd.read_csv(r".\Data\Final_Adjusted_Employee_Scheduling_Data.csv")
# data=df.copy()
# st.title("AttendAI")

# Streamlit UI
st.set_page_config(page_title="AttendAI", layout="wide")
st.markdown("""
    <h1 style='text-align: center;'>AttendAI</h1>
    """, unsafe_allow_html=True)
# Data preview
with st.expander("Data Preview"):
    st.write(data.tail(5))

query = st.text_area("Ask your data a question:")

if query:
    llm= OpenAI(api_token="YOURAPIKEY",model="gpt-4o")
    query_engine = SmartDataframe(data, config={"llm": llm})
    answer = query_engine.chat(query)
    st.write("### Chatbot Response:")
    st.write(answer)

data['Date'] = pd.to_datetime(data['Date'])
daily_counts = data.groupby('Date').size().reset_index(name='Employees_Needed')
daily_counts.set_index('Date', inplace=True)

# Forecast future values
def forecast_future(model, recent_data, scaler, days_to_predict=14, n_steps=7):
    predictions = []
    current_input = recent_data[-n_steps:].reshape(1, n_steps, 1)
    for _ in range(days_to_predict):
        next_pred = model.predict(current_input, verbose=0)[0, 0]
        predictions.append(next_pred)
        current_input = np.append(current_input[:, 1:, :], [[[next_pred]]], axis=1)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Streamlit UI
st.title('Employee Scheduling Forecast')

# Slider to select forecast length
days = st.slider('Days to forecast:', 7, 28, 14)

# Generate forecast
scaled_data = scaler.transform(daily_counts)
forecast = forecast_future(model, scaled_data, scaler, days_to_predict=days)

# Display forecast
forecast_dates = pd.date_range(start=daily_counts.index[-1] + pd.Timedelta(days=1), periods=days)
forecast_df = pd.DataFrame(forecast, index=forecast_dates, columns=['Employees Needed'])

st.subheader('Forecasted Employees Needed')
st.write(forecast_df)

# Plot forecast
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(daily_counts.index, daily_counts, label='Historical Data')
ax.plot(forecast_dates, forecast, label='Forecast', linestyle='--', color='orange')
ax.set_title('Employee Scheduling Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Employees Needed')
ax.legend()
st.pyplot(fig)