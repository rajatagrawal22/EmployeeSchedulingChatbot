import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

# Load and preprocess data
data = pd.read_csv(r".\Data\Final_Adjusted_Employee_Scheduling_Data.csv")
data['Date'] = pd.to_datetime(data['Date'])

# Aggregate data to get daily employee counts
daily_counts = data.groupby('Date').size().reset_index(name='Employees_Needed')
daily_counts.set_index('Date', inplace=True)

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily_counts)

# Prepare data for LSTM
X, y = [], []
n_steps = 7
for i in range(n_steps, len(scaled_data)):
    X.append(scaled_data[i-n_steps:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-test split
X_train, X_test = X[:-30], X[-30:]
y_train, y_test = y[:-30], y[-30:]

# Define and fit LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, 1)))
model.add(LSTM(25, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Evaluate model
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_rescaled, predictions)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions))

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Dump model and scaler
model.save('lstm_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)