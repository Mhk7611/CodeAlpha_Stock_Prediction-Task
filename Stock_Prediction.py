# Step 1: Install necessary libraries
# You can install them via Jupyter using the following commands
# !pip install yfinance numpy pandas matplotlib scikit-learn keras tensorflow

# Step 2: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Step 3: Download Stock Data (Using Tesla as an example)
stock_symbol = 'TSLA'  # You can change this to any company's stock symbol
stock_data = yf.download(stock_symbol, start='2010-01-01', end='2023-01-01')

# Step 4: Data Preprocessing
# Using only 'Close' price for prediction
data = stock_data[['Close']]

# Scale the data using MinMaxScaler to normalize it between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create datasets for LSTM model (using 60 previous days to predict the next day)
def create_dataset(data, time_step=60):
    X = []
    y = []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])  # Last 60 days' prices
        y.append(data[i, 0])  # Next day's price (target)
    return np.array(X), np.array(y)

# Split data into train and test sets
training_data_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:training_data_len]
test_data = scaled_data[training_data_len:]

# Prepare training and test datasets
X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# Reshape the data for LSTM (LSTM expects input of shape [samples, time_steps, features])
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 5: Build the LSTM Model
model = Sequential()

# Add first LSTM layer with dropout to avoid overfitting
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

# Add second LSTM layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Add the output layer
model.add(Dense(units=1))  # Predicting the next stock price

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print model summary
model.summary()

# Step 6: Train the Model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Step 7: Predict the Stock Prices using the Model
predictions = model.predict(X_test)

# Step 8: Reverse scaling for predictions and actual values
predictions = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 9: Plot the Results (Predicted vs Real Stock Price)
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, color='blue', label=f'Real {stock_symbol} Stock Price')
plt.plot(predictions, color='red', label=f'Predicted {stock_symbol} Stock Price')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()

# Step 10: Calculate the Model's Performance (Mean Squared Error)
mse = mean_squared_error(y_test_rescaled, predictions)
print(f'Mean Squared Error: {mse}')
import tensorflow as tf
print(tf.__version__)

