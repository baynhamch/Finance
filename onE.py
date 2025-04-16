from binance.client import Client
import os
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import joblib
import ta

# ----------------------- Binance Client -----------------------
class BinanceUSClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        self.secret_key = os.getenv("SECRET_KEY")
        self.client = Client(self.api_key, self.secret_key, tld='us')

    def get_price(self, symbol="BTCUSDT"):
        tickers = self.client.get_all_tickers()
        for ticker in tickers:
            if ticker["symbol"] == symbol:
                return float(ticker["price"])
        return None

    def get_usdt_balance(self):
        balance = self.client.get_asset_balance(asset='USDT')
        return float(balance['free']) if balance else 0.0

    def get_historical_data(self, symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_5MINUTE, lookback="3 day ago UTC"):
        klines = self.client.get_historical_klines(symbol, interval, lookback)
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        return df

# ----------------------- Logger -----------------------
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open("onE_log_v2.txt", "a") as file:
        file.write(full_message + "\n")

# ----------------------- Position Manager -----------------------
class PositionManager:
    def __init__(self):
        self.in_position = False
        self.entry_price = None
        self.take_profit = None
        self.stop_loss = None
        self.quantity = None

    def open_position(self, price, quantity, tp_multiplier=1.015, sl_multiplier=0.9925):
        self.in_position = True
        self.entry_price = price
        self.quantity = quantity
        self.take_profit = price * tp_multiplier
        self.stop_loss = price * sl_multiplier
        log(f"🟢 Entered position at ${price:.2f} | TP: ${self.take_profit:.2f} | SL: ${self.stop_loss:.2f}")

    def should_close_position(self, current_price):
        if current_price >= self.take_profit:
            log(f"✅ Take Profit Hit at ${current_price:.2f}")
            return True
        if current_price <= self.stop_loss:
            log(f"🛑 Stop Loss Hit at ${current_price:.2f}")
            return True
        return False

    def close_position(self):
        log("🔴 Closing position.")
        self.in_position = False
        self.entry_price = None
        self.take_profit = None
        self.stop_loss = None
        self.quantity = None

# ----------------------- Features & Label -----------------------
def calculate_indicators(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['MACD'] = ta.trend.MACD(df['close']).macd()
    bb = ta.volatility.BollingerBands(df['close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    return df.dropna()

def add_target(df, horizon=3):
    df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
    df['target'] = 0  # Hold
    df.loc[df['future_return'] > 0.002, 'target'] = 1  # Buy
    df.loc[df['future_return'] < -0.002, 'target'] = -1  # Sell
    return df.dropna()

# ----------------------- Model Training -----------------------
def train_or_load_model(df, model_path='model2.pkl'):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    X = df[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower']]
    y = df['target']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

# ----------------------- Prediction -----------------------
def predict(df, model):
    latest = df.iloc[-1]
    X = latest[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower']].values.reshape(1, -1)
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    return pred, max(proba)

# ----------------------- Trading Logic -----------------------
position = PositionManager()
client = BinanceUSClient()
BUY_BUFFER = 0.997
SELL_BUFFER = 1.005
CONFIDENCE_THRESHOLD = 0.7

last_trade_time = None

def can_trade():
    global last_trade_time
    if not last_trade_time or (datetime.now() - last_trade_time).seconds > 300:
        last_trade_time = datetime.now()
        return True
    return False

def main():
    log("🔁 Running Project onE...")
    if not can_trade():
        log("⏳ Trade cooldown active. Skipping.")
        return

    live_price = client.get_price()
    usdt_balance = client.get_usdt_balance()
    log(f"💰 USDT Balance: ${usdt_balance:.2f}")

    df = client.get_historical_data()
    df = calculate_indicators(df)
    df = add_target(df)
    model = train_or_load_model(df)

    prediction, confidence = predict(df, model)
    log(f"Model Prediction: {prediction} | Confidence: {confidence:.2f}")

    if confidence < CONFIDENCE_THRESHOLD:
        log("⚠️ Low confidence. Holding.")
        return

    if not position.in_position:
        if prediction == 1:
            trade_usdt = round(usdt_balance * 0.20, 2)
            btc_amount = round(trade_usdt / live_price, 6)
            try:
                order = client.client.create_order(
                    symbol='BTCUSDT',
                    side='BUY',
                    type='MARKET',
                    quoteOrderQty=trade_usdt
                )
                position.open_position(live_price, btc_amount)
                log(f"✅ BUY ORDER: {order}")
            except Exception as e:
                log(f"❌ Failed to BUY: {e}")
        else:
            log("💤 No BUY signal. Waiting...")
    elif prediction == -1 and position.should_close_position(live_price):
        try:
            order = client.client.create_order(
                symbol='BTCUSDT',
                side='SELL',
                type='MARKET',
                quantity=position.quantity
            )
            position.close_position()
            log(f"✅ SELL ORDER: {order}")
        except Exception as e:
            log(f"❌ Failed to SELL: {e}")
    else:
        log("📊 In position. Holding...")

# ----------------------- Loop -----------------------
def run_forever():
    while True:
        main()
        log("⏱ Waiting 1 minute...")
        time.sleep(60)

if __name__ == "__main__":
    run_forever()
