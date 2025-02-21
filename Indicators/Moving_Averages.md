## Moving Averages (SMA & EMA)
Simple Moving Average (SMA) and Exponential Moving Average (EMA) are commonly used trend-following indicators.
```python
import pandas as pd

def calculate_sma(data, period=20):
    """Calculate Simple Moving Average (SMA)"""
    return data['Close'].rolling(window=period).mean()

def calculate_ema(data, period=20):
    """Calculate Exponential Moving Average (EMA)"""
    return data['Close'].ewm(span=period, adjust=False).mean()
```

## Relative Strength Index (RSI)
``` python
def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

## Moving Average Convergence Divergence (MACD)
```python
def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    """Calculate MACD and Signal Line"""
    short_ema = data['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal
```

## Bollinger Bands
```python
def calculate_bollinger_bands(data, period=20, num_std=2):
    """Calculate Bollinger Bands (Upper & Lower)"""
    sma = calculate_sma(data, period)
    std = data['Close'].rolling(window=period).std()
    
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    
    return upper_band, lower_band
```

## Average True Range (ATR)
```python
def calculate_atr(data, period=14):
    """Calculate Average True Range (ATR)"""
    data['High-Low'] = data['High'] - data['Low']
    data['High-Close'] = abs(data['High'] - data['Close'].shift())
    data['Low-Close'] = abs(data['Low'] - data['Close'].shift())
    
    tr = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr
```

## Stochastic Oscillator

```python
def calculate_stochastic_oscillator(data, period=14):
    """Calculate Stochastic Oscillator (%K and %D)"""
    lowest_low = data['Low'].rolling(window=period).min()
    highest_high = data['High'].rolling(window=period).max()
    
    k_percent = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=3).mean()  # Signal Line
    
    return k_percent, d_percent
```

## On-Balance Volume (OBV)
```python
def calculate_obv(data):
    """Calculate On-Balance Volume (OBV)"""
    obv = (data['Volume'] * ((data['Close'].diff() > 0) * 2 - 1)).fillna(0).cumsum()
    return obv
```

## Example
```python
import yfinance as yf

# Fetch stock data
ticker = "AAPL"
df = yf.download(ticker, start="2023-01-01", end="2024-01-01")

# Compute indicators
df['SMA_20'] = calculate_sma(df, 20)
df['EMA_20'] = calculate_ema(df, 20)
df['RSI_14'] = calculate_rsi(df, 14)
df['MACD'], df['Signal'] = calculate_macd(df)
df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(df)
df['ATR_14'] = calculate_atr(df)
df['%K'], df['%D'] = calculate_stochastic_oscillator(df)
df['OBV'] = calculate_obv(df)

# Display results
print(df.tail())
```
