# Employee Scheduling Analysis Chatbot

This is an AI-powered Streamlit chatbot for analyzing employee scheduling data. It provides attendance insights, overtime tracking, and predictive staffing using pre-trained LSTM models.
# Project Overview

This project leverages machine learning and AI to help organizations manage employee schedules more efficiently. The chatbot can:
- Analyze attendance patterns and identify anomalies.
- Track overtime and provide summaries for management.
- Predict staffing needs using Long Short-Term Memory (LSTM) models.
- Offer a conversational interface for managers and HR teams to query scheduling data easily.

The project is built on Python, with a focus on Streamlit for the interactive UI and TensorFlow/Keras for the predictive models.
---

## 🚀 Features
- Natural language queries on employee shift data
- Predict daily, weekly, and monthly staffing needs
- Visualize time series trends
- Attendance Analysis

-Visualizes employee attendance trends.

--Detects irregular patterns, late arrivals, and absenteeism.

Overtime Tracking

Calculates total overtime per employee.

Provides reports for payroll and workforce planning.

Predictive Staffing

Uses LSTM models to forecast staffing requirements.

Helps optimize scheduling to reduce understaffing or overstaffing.

Interactive Chatbot Interface

Ask natural language questions about schedules, overtime, or predictions.

Immediate, visual, and text-based responses.
## 🧠 Prerequisites
- Python 3.9+
- OpenAI API key (for PandasAI integration)

## 🛠️ Usage
### 1. Train and Save Models
Run the following script to train LSTM models for daily, weekly, and monthly forecasting:
```bash
python lstm_model.py
```

### 2. Launch the Chatbot App
Start the Streamlit application:
```bash
streamlit run app.py
