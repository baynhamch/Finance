from binance.client import Client
import os, time, joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import ta

# ----------------------- Binance Client -----------------------
class BinanceUSClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        self.secret_key = os.getenv("SECRET_KEY")
        self.client = Client(self.api_key, self.secret_key, tld='us')

    def get_price(self, symbol="BTCUSDT"):
        return float(self.client.get_symbol_ticker(symbol=symbol)["price"])

    def get_usdt_balance(self):
        balance = self.client.get_asset_balance(asset='USDT')
        return float(balance['free']) if balance else 0.0

    def get_order_book_stats(self, symbol="BTCUSDT"):
        depth = self.client.get_order_book(symbol=symbol)
        bid_volume = sum(float(bid[1]) for bid in depth['bids'])
        ask_volume = sum(float(ask[1]) for ask in depth['asks'])
        return bid_volume, ask_volume

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
    with open("onE_log_v5.txt", "a") as f:
        f.write(full_message + "\n")

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
        self.in_position = False
        self.entry_price = None
        self.take_profit = None
        self.stop_loss = None
        self.quantity = None
        log("🔴 Closed position.")

# ----------------------- Indicators -----------------------
def calculate_indicators(df, bid_volume, ask_volume):
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['MACD'] = ta.trend.MACD(df['close']).macd()
    bb = ta.volatility.BollingerBands(df['close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['OBI'] = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
    return df.dropna()

def add_target(df, horizon=3):
    df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
    df['target'] = 0
    df.loc[df['future_return'] > 0.002, 'target'] = 1
    df.loc[df['future_return'] < -0.002, 'target'] = -1
    return df.dropna()

# ----------------------- Multi-Timeframe Trend Confirmation -----------------------
def confirm_trend(df_15m):
    df_15m['SMA_50'] = ta.trend.sma_indicator(df_15m['close'], window=50)
    return df_15m['close'].iloc[-1] > df_15m['SMA_50'].iloc[-1]

# ----------------------- Model Training -----------------------
def train_or_load_model(df, model_path='model_v5.pkl'):
    features = ['SMA_20','SMA_50','RSI','MACD','BB_upper','BB_lower','ATR','Stoch_K','Stoch_D','VWAP','OBI','volume_zscore']
    if os.path.exists(model_path):
        return joblib.load(model_path)
    X, y = df[features], df['target']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

# ----------------------- Prediction -----------------------
def predict(df, model):
    features = ['SMA_20','SMA_50','RSI','MACD','BB_upper','BB_lower','ATR','Stoch_K','Stoch_D','VWAP','OBI','volume_zscore']
    latest = df.iloc[-1][features].values.reshape(1, -1)
    proba = model.predict_proba(latest)[0]
    pred = model.predict(latest)[0]
    return pred, max(proba)

# ----------------------- Main Logic -----------------------
position = PositionManager()
client = BinanceUSClient()
last_trade_time = None
CONFIDENCE_THRESHOLD = 0.7

def can_trade():
    global last_trade_time
    if not last_trade_time or (datetime.now() - last_trade_time).seconds > 300:
        last_trade_time = datetime.now()
        return True
    return False

def main():
    log("🔁 Running Project onE...")
    if not can_trade():
        log("⏳ Cooldown. Skipping.")
        return

    live_price = client.get_price()
    usdt_balance = client.get_usdt_balance()
    bid_volume, ask_volume = client.get_order_book_stats()

    df = client.get_historical_data()
    df = calculate_indicators(df, bid_volume, ask_volume)
    df = add_target(df)

    df_15m = client.get_historical_data(interval=Client.KLINE_INTERVAL_15MINUTE, lookback="3 day ago UTC")
    trend_ok = confirm_trend(df_15m)

    model = train_or_load_model(df)
    prediction, confidence = predict(df, model)
    log(f"Prediction: {prediction}, Confidence: {confidence:.2f}, Trend OK: {trend_ok}")

    if confidence < CONFIDENCE_THRESHOLD or not trend_ok:
        log("⚠️ Not confident or weak trend. Holding.")
        return

    if not position.in_position and prediction == 1:
        trade_usdt = round(usdt_balance * 0.20, 2)
        btc_amount = round(trade_usdt / live_price, 6)
        try:
            order = client.client.create_order(symbol='BTCUSDT', side='BUY', type='MARKET', quoteOrderQty=trade_usdt)
            position.open_position(live_price, btc_amount)
            log(f"✅ BUY ORDER: {order}")
        except Exception as e:
            log(f"❌ BUY failed: {e}")
    elif prediction == -1 and position.should_close_position(live_price):
        try:
            order = client.client.create_order(symbol='BTCUSDT', side='SELL', type='MARKET', quantity=position.quantity)
            position.close_position()
            log(f"✅ SELL ORDER: {order}")
        except Exception as e:
            log(f"❌ SELL failed: {e}")
    else:
        log("📊 In position. Holding...")

# ----------------------- Loop -----------------------
def run_forever():
    while True:
        main()
        log("⏱ Sleeping 5 minutes...\n")
        time.sleep(60)

if __name__ == "__main__":
    run_forever()
