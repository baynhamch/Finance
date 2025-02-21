# AI-Based Trading Bot Using LSTM in Python

This guide walks you through building an **AI-driven algorithmic trading bot** using a **Long Short-Term Memory (LSTM) neural network** to predict stock prices and trade using the **Webull API**.

## **1. Setup Environment**

### **Install Dependencies**
```bash
pip install tensorflow keras pandas numpy matplotlib scikit-learn yfinance webull
```

---

## **2. Fetch Historical Stock Data**

We use `yfinance` to retrieve historical stock price data.

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data for a specific stock (e.g., AAPL)
ticker = "AAPL"
df = yf.download(ticker, start="2010-01-01", end="2024-01-01")

# Keep only 'Close' prices
df = df[['Close']]
df.dropna(inplace=True)

# Plot the stock price
plt.figure(figsize=(10,5))
plt.plot(df.Close, label=f"{ticker} Stock Price")
plt.legend()
plt.show()
```

---

## **3. Data Preprocessing**

### **Normalize Data**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=['Close'])
```

### **Create Sequences for LSTM**
```python
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

time_steps = 60
X, y = create_sequences(df_scaled.values, time_steps)

# Reshape for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
```

---

## **4. Build the LSTM Model**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X, y, epochs=25, batch_size=32)
```

---

## **5. Predict Next Day's Price**

```python
# Prepare latest data for prediction
test_data = df_scaled[-time_steps:].values
test_data = np.reshape(test_data, (1, time_steps, 1))

# Predict
predicted_price = model.predict(test_data)
predicted_price = scaler.inverse_transform(predicted_price)

print(f"Predicted Stock Price: {predicted_price[0][0]}")
```

---

## **6. Backtesting Model Performance**

```python
train_size = int(len(df) * 0.8)
train, test = df_scaled[:train_size], df_scaled[train_size:]

X_test, y_test = create_sequences(test, time_steps)
predicted_prices = model.predict(X_test)

# Convert back to original scale
predicted_prices = scaler.inverse_transform(predicted_prices)
y_test = scaler.inverse_transform(y_test)

# Plot results
plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual Price")
plt.plot(predicted_prices, label="Predicted Price", linestyle='dashed')
plt.legend()
plt.show()
```

---

## **7. Deploy for Live Trading with Webull**

### **Authenticate Webull API**
```python
from webull import webull

wb = webull()
wb.login(username='your_email', password='your_password')
```

### **Fetch Live Data and Execute Trades**
```python
latest_data = wb.get_bars(ticker, interval='d1', count=60)
latest_data = pd.DataFrame(latest_data)[['close']]

# Normalize & Predict
latest_data_scaled = scaler.transform(latest_data.values.reshape(-1,1))
latest_data_scaled = np.reshape(latest_data_scaled, (1, time_steps, 1))

predicted_price = model.predict(latest_data_scaled)
predicted_price = scaler.inverse_transform(predicted_price)

print(f"Predicted Next Price: {predicted_price[0][0]}")

# Example Trade Decision
if predicted_price[0][0] > latest_data.iloc[-1]['close']:
    wb.place_order(stock=ticker, price=0, action='BUY', orderType='MKT', enforce='GTC', qty=10)
```

---

## **8. Next Steps**
- **Optimize Hyperparameters**: Adjust LSTM layers, epochs, batch size.
- **Use More Features**: Include volume, RSI, MACD.
- **Deploy on Cloud**: Run script on AWS for 24/7 trading.

---

### **📌 Notes**
- **Paper Trade First!** Always test on a paper trading account before deploying real money.
- **Risk Management**: Implement stop losses and take profits.
- **Monitor Performance**: Log trades and analyze the model’s accuracy.
