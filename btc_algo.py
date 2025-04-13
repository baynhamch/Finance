
import pandas as pd
import numpy as np
import ccxt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# -------- SETTINGS --------
BUY_BUFFER = 0.995
SELL_BUFFER = 1.005


# -------- Get Live Binance Price --------
def get_live_btc_price():
    binance = ccxt.binanceus()
    ticker = binance.fetch_ticker('BTC/USDT')
    return ticker['last']


# -------- Generate Fake BTC Data for Indicators --------
def generate_fake_btc_data(days=200):
    np.random.seed(42)
    base_price = 30000
    prices = base_price + np.random.randn(days).cumsum()
    volumes = np.random.randint(1000, 10000, size=days)

    df = pd.DataFrame({
        'timestamp': [datetime.now() - timedelta(days=i) for i in reversed(range(days))],
        'close': prices,
        'open': prices + np.random.randint(-100, 100, size=days),
        'high': prices + np.random.randint(0, 150, size=days),
        'low': prices - np.random.randint(0, 150, size=days),
        'volume': volumes
    })
    df.set_index('timestamp', inplace=True)
    return df


# -------- Calculate Indicators --------
def calculate_indicators(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df.dropna()


# -------- Generate Signals --------
def generate_signals(df):
    df['Buy_Signal'] = (df['SMA_20'] > df['SMA_50']) & (df['RSI'] < 30)
    df['Sell_Signal'] = (df['SMA_20'] < df['SMA_50']) & (df['RSI'] > 70)
    df['Hold_Signal'] = ~(df['Buy_Signal'] | df['Sell_Signal'])
    return df


# -------- Train Model --------
def train_model(df):
    X = df[['SMA_20', 'SMA_50', 'RSI']]
    y = df['Buy_Signal'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# -------- Predict Trade --------
def predict_trade(df, model, live_price):
    df.iloc[-1, df.columns.get_loc('close')] = live_price  # Use live price in indicators

    latest = df.iloc[-1]
    rec_buy = live_price * BUY_BUFFER
    rec_sell = live_price * SELL_BUFFER

    input_data = latest[['SMA_20', 'SMA_50', 'RSI']].values.reshape(1, -1)

    if latest['Buy_Signal']:
        return (
            f"📈 BUY SIGNAL\n"
            f"Live Price:         ${live_price:.2f}\n"
            f"🎯 Suggested Buy:   ${rec_buy:.2f}\n"
            f"💰 Target Sell:     ${rec_sell:.2f}"
        )
    elif latest['Sell_Signal']:
        return (
            f"📉 SELL SIGNAL\n"
            f"Live Price:         ${live_price:.2f}\n"
            f"🛑 Suggested Sell:  ${rec_sell:.2f}\n"
            f"🔄 Consider Buy:    ${rec_buy:.2f}"
        )
    else:
        return (
            f"⚖️ HOLD SIGNAL\n"
            f"Live Price:         ${live_price:.2f}\n"
            f"🎯 Suggested Buy:   ${rec_buy:.2f}\n"
            f"💰 Target Sell:     ${rec_sell:.2f}"
        )

# -------- MAIN --------
def main():
    live_price = get_live_btc_price()
    df = generate_fake_btc_data()
    df = calculate_indicators(df)
    df = generate_signals(df)
    model = train_model(df)
    decision = predict_trade(df, model, live_price)
    print(decision)


# -------- RUN --------
if __name__ == "__main__":
    main()
