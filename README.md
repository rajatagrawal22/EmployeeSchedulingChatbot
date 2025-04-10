# Employee Scheduling Analysis Chatbot

This is an AI-powered Streamlit chatbot for analyzing employee scheduling data. It provides attendance insights, overtime tracking, and predictive staffing using pre-trained LSTM models.

---

## 🚀 Features
- Natural language queries on employee shift data
- Predict daily, weekly, and monthly staffing needs
- Visualize time series trends
## 🧠 Prerequisites
- Python 3.9+
- OpenAI API key (for PandasAI integration)

- 
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
