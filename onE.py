from binance.client import Client
import os
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

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

    def get_historical_data(self, symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1HOUR, lookback="30 days ago UTC"):
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
    with open("onE_log.txt", "a") as file:
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

# ----------------------- Indicators & Model -----------------------
def calculate_indicators(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df.dropna()

def generate_signals(df):
    df['Buy_Signal'] = (df['SMA_20'] > df['SMA_50']) & (df['RSI'] < 30)
    df['Sell_Signal'] = (df['SMA_20'] < df['SMA_50']) & (df['RSI'] > 70)
    df['Hold_Signal'] = ~(df['Buy_Signal'] | df['Sell_Signal'])
    return df

def train_model(df):
    X = df[['SMA_20', 'SMA_50', 'RSI']]
    y = df['Buy_Signal'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# ----------------------- Prediction -----------------------
BUY_BUFFER = 0.995
SELL_BUFFER = 1.005

def predict_trade(df, model, live_price):
    df.iloc[-1, df.columns.get_loc('close')] = live_price
    latest = df.iloc[-1]
    rec_buy = live_price * BUY_BUFFER
    rec_sell = live_price * SELL_BUFFER
    input_data = latest[['SMA_20', 'SMA_50', 'RSI']].values.reshape(1, -1)
    prediction = model.predict(input_data)[0]
    if latest['Buy_Signal']:
        return f"📈 BUY SIGNAL\nLive Price: ${live_price:.2f}\nBuy Below: ${rec_buy:.2f}\nTarget Sell: ${rec_sell:.2f}"
    elif latest['Sell_Signal']:
        return f"📉 SELL SIGNAL\nLive Price: ${live_price:.2f}\nSell Above: ${rec_sell:.2f}\nBuy Target: ${rec_buy:.2f}"
    else:
        return f"⚖️ HOLD\nLive Price: ${live_price:.2f}\nBuy Below: ${rec_buy:.2f}\nTarget Sell: ${rec_sell:.2f}"

# ----------------------- Trading Logic -----------------------
position = PositionManager()
client = BinanceUSClient()

def main():
    log("🔁 Running Auto BTC Trader...")
    live_price = client.get_price()
    usdt_balance = client.get_usdt_balance()
    log(f"💰 USDT Balance: ${usdt_balance:.2f}")
    if usdt_balance < 10:
        log("❌ Not enough USDT to trade. Skipping.")
        return

    df = client.get_historical_data()
    df = calculate_indicators(df)
    df = generate_signals(df)
    model = train_model(df)
    decision = predict_trade(df, model, live_price)
    log(decision)

    if not position.in_position:
        if "BUY SIGNAL" in decision:
            trade_usdt = round(usdt_balance * 0.20, 2)
            btc_price = live_price
            btc_amount = round(trade_usdt / btc_price, 6)

            try:
                order = client.client.create_order(
                    symbol='BTCUSDT',
                    side='BUY',
                    type='MARKET',
                    quoteOrderQty=trade_usdt
                )
                position.open_position(btc_price, btc_amount)
                log(f"✅ BUY ORDER PLACED: {order}")
            except Exception as e:
                log(f"❌ Failed to BUY: {e}")
        else:
            log("💤 No BUY signal. Waiting...")
    else:
        if position.should_close_position(live_price):
            try:
                order = client.client.create_order(
                    symbol='BTCUSDT',
                    side='SELL',
                    type='MARKET',
                    quantity=position.quantity
                )
                position.close_position()
                log(f"✅ SELL ORDER PLACED: {order}")
            except Exception as e:
                log(f"❌ Failed to SELL: {e}")
        else:
            log("📊 In position. Holding...")

# ----------------------- Loop It -----------------------
def run_forever():
    while True:
        main()
        log("⏱ Waiting 30 minutes...\n")
        time.sleep(1800)

if __name__ == "__main__":
    run_forever()
