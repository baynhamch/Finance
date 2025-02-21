import yfinance as yf
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from twilio.rest import Client

# Function to fetch stock data
def fetch_stock_data(stocks):
    """Download historical stock data for multiple stocks."""
    today = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(stocks, start="2020-01-01", end=today)
    return df


# Function to calculate indicators
def calculate_indicators(df):
    """Calculate SMA and RSI for a given stock dataframe."""
    df['SMA_20'] = df['Close'].rolling(window=20).mean()  # 20-day SMA
    df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50-day SMA

    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df.dropna()


# Function to generate trading signals
# def generate_signals(df):
#     """Generate Buy/Sell signals."""
#     df = df.copy()  # Ensure we're modifying a copy
#     df.loc[:, 'Buy_Signal'] = (df['SMA_20'] > df['SMA_50']) & (df['RSI'] < 30)
#     df.loc[:, 'Sell_Signal'] = (df['SMA_20'] < df['SMA_50']) & (df['RSI'] > 70)
#     return df
# def generate_signals(df):
#     """Generate Buy, Hold, and Sell signals."""
#     df = df.copy()  # Ensure we're modifying a copy
#
#     # Buy if SMA_20 > SMA_50 and RSI < 30 (Oversold)
#     df.loc[:, 'Buy_Signal'] = (df['SMA_20'] > df['SMA_50']) & (df['RSI'] < 30)
#
#     # Sell if SMA_20 < SMA_50 and RSI > 70 (Overbought)
#     df.loc[:, 'Sell_Signal'] = (df['SMA_20'] < df['SMA_50']) & (df['RSI'] > 70)
#
#     # Hold if neither Buy nor Sell signals are triggered
#     df.loc[:, 'Hold_Signal'] = ~(df['Buy_Signal'] | df['Sell_Signal'])
#     df.loc[:, 'Buy_Signal'] |= (df['MACD'] > df['Signal_Line']) & (df['RSI'] < 40)
#     df.loc[:, 'Sell_Signal'] |= (df['MACD'] < df['Signal_Line']) & (df['RSI'] > 60)
#
#     return df
def generate_signals(df):
    """Generate Buy, Hold, and Sell signals with more indicators."""
    df = df.copy()  # Ensure we're modifying a copy

    # MACD Calculation
    df['MACD'] = df['SMA_20'] - df['SMA_50']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['Middle_Band'] = df['SMA_20']
    df['Upper_Band'] = df['SMA_20'] + (df['Close'].rolling(20).std() * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['Close'].rolling(20).std() * 2)

    # VWAP Calculation (only if 'Volume' column exists)
    if 'Volume' in df.columns:
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    else:
        df['VWAP'] = float('nan')  # Assign NaN if Volume is missing

    # Buy Signal Conditions
    df.loc[:, 'Buy_Signal'] = (
        ((df['SMA_20'] > df['SMA_50']) & (df['RSI'] < 30)) |
        ((df['MACD'] > df['Signal_Line']) & (df['RSI'] < 40)) |
        ((df['Close'] < df['Lower_Band']) & (df['RSI'] < 30))
    )

    # Add VWAP-based buy signal only if VWAP exists
    if 'VWAP' in df.columns and df['VWAP'].notna().all():
        df.loc[:, 'Buy_Signal'] |= (df['Close'] > df['VWAP']) & (df['RSI'] < 40)

    # Sell Signal Conditions
    df.loc[:, 'Sell_Signal'] = (
        ((df['SMA_20'] < df['SMA_50']) & (df['RSI'] > 70)) |
        ((df['MACD'] < df['Signal_Line']) & (df['RSI'] > 60)) |
        ((df['Close'] > df['Upper_Band']) & (df['RSI'] > 70))
    )

    # Hold Signal (if neither Buy nor Sell conditions are met)
    df.loc[:, 'Hold_Signal'] = ~(df['Buy_Signal'] | df['Sell_Signal'])

    return df  # ✅ Ensure the function returns df

# Function to process stocks (apply indicators & signals)
def process_stocks(df, stocks):
    """Apply indicators & generate signals for each stock."""
    processed_data = {}
    for stock in stocks:
        stock_df = pd.DataFrame(df['Close'][stock]).rename(columns={stock: 'Close'})
        processed_data[stock] = generate_signals(calculate_indicators(stock_df))
    return processed_data


# Function to train ML model
def train_ml_model(all_data):
    """Train a Random Forest model for stock predictions."""
    X = all_data[['SMA_20', 'SMA_50', 'RSI']].dropna()
    y = all_data['Buy_Signal'].astype(int)  # 1 = Buy, 0 = Hold

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"✅ Model Accuracy: {accuracy:.2f}")

    return model, X


# Function to predict buy/sell signals
# def predict_trades(model, X, stocks, processed_data):
#     """Predict Buy/Sell/Hold signals for each stock and provide reasoning."""
#     latest_data = X.tail(len(stocks))  # Get latest data for all stocks
#     predicted_signals = model.predict(latest_data)
#
#     trade_signals = {}  # Store trade decisions
#     for stock, signal in zip(stocks, predicted_signals):
#         stock_data = processed_data[stock].iloc[-1]  # Get latest row of indicators
#
#         # Determine reasoning
#         if signal == 1:
#             reason = f"SMA_20 ({stock_data['SMA_20']:.2f}) > SMA_50 ({stock_data['SMA_50']:.2f}) and RSI ({stock_data['RSI']:.2f}) < 30 (oversold)"
#             trade_signals[stock] = f"📈 BUY Signal for {stock} - {reason}"
#         else:
#             reason = f"SMA_20 ({stock_data['SMA_20']:.2f}) <= SMA_50 ({stock_data['SMA_50']:.2f}) or RSI ({stock_data['RSI']:.2f}) > 30 (neutral/overbought)"
#             trade_signals[stock] = f"📉 HOLD/SELL Signal for {stock} - {reason}"
#
#     return trade_signals

# Function to predict buy/hold/sell signals
def predict_trades(model, X, stocks, processed_data):
    """Predict Buy, Hold, or Sell signals for each stock and provide reasoning."""
    latest_data = X.tail(len(stocks))  # Get latest data for all stocks
    predicted_signals = model.predict(latest_data)

    trade_signals = {}  # Store trade decisions
    for stock, signal in zip(stocks, predicted_signals):
        stock_data = processed_data[stock].iloc[-1]  # Get latest row of indicators

        # Determine Buy, Hold, or Sell Reasoning
        if stock_data['Buy_Signal']:
            reason = f"SMA_20 ({stock_data['SMA_20']:.2f}) > SMA_50 ({stock_data['SMA_50']:.2f}) and RSI ({stock_data['RSI']:.2f}) < 30 (oversold)"
            trade_signals[stock] = f"📈 BUY Signal for {stock} - {reason}"

        elif stock_data['Sell_Signal']:
            reason = f"SMA_20 ({stock_data['SMA_20']:.2f}) < SMA_50 ({stock_data['SMA_50']:.2f}) and RSI ({stock_data['RSI']:.2f}) > 70 (overbought)"
            trade_signals[stock] = f"📉 SELL Signal for {stock} - {reason}"

        else:
            reason = f"SMA_20 ({stock_data['SMA_20']:.2f}) and SMA_50 ({stock_data['SMA_50']:.2f}) show no clear trend, RSI ({stock_data['RSI']:.2f}) is neutral"
            trade_signals[stock] = f"⚖️ HOLD Signal for {stock} - {reason}"

    return trade_signals
def send_message(message_body, recipient_number, twilio_number, account_sid, auth_token):
    """
       Sends an SMS using Twilio API.

       Parameters:
       - message_body (str): The message to send.
       - recipient_number (str): The phone number to send the message to.
       - twilio_number (str): Your Twilio phone number.
       - account_sid (str): Your Twilio Account SID.
       - auth_token (str): Your Twilio Auth Token.

       Returns:
       - str: Message SID if successful.
       """
    try:
        # Create Twilio client
        client = Client(account_sid, auth_token)

        # Send SMS
        message = client.messages.create(
            body=message_body,
            from_=twilio_number,
            to=recipient_number
        )

        print(f"✅ SMS sent successfully! Message SID: {message.sid}")
        return message.sid

    except Exception as e:
        print(f"❌ Failed to send SMS: {e}")
        return None


# Main function
def main():
    stocks = ["AAPL", "TSLA", "NVDA", "META", "SPY", "GOOG", "AMZN", "BABA", "V", "JNJ", "HD", "JPM", "PG", "MA", "BTC", "MCD", "NFLX", "BAC", "KO", 'TJX', "T"]
    # stocks = [
    #     "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK.A", "TSLA", "JNJ", "V",
    #     "UNH", "WMT", "JPM", "MA", "PG", "XOM", "HD", "BAC", "KO", "PFE",
    #     "DIS", "CSCO", "PEP", "VZ", "ADBE", "NFLX", "INTC", "CMCSA", "MRK", "T",
    #     "ABT", "NKE", "CRM", "ORCL", "CVX", "MCD", "WFC", "ACN", "DHR", "MDT",
    #     "COST", "LLY", "AVGO", "QCOM", "TXN", "NEE", "HON", "PM", "BMY", "IBM",
    #     "SBUX", "AMGN", "MMM", "LIN", "GE", "LOW", "UPS", "MS", "UNP", "RTX",
    #     "INTU", "BA", "CAT", "GS", "BLK", "AXP", "SPGI", "PLD", "MDLZ", "SYK",
    #     "ISRG", "TMO", "AMT", "BKNG", "DE", "ADP", "GILD", "NOW", "MO", "FIS",
    #     "CI", "CB", "USB", "SCHW", "ZTS", "C", "LMT", "BDX", "DUK", "SO",
    #     "TJX", "PNC", "MMC", "CCI", "APD", "EL", "ADI", "ITW", "NSC", "EW"
    # ]
    # Step 1: Fetch stock data
    df = fetch_stock_data(stocks)

    # Step 2: Process stocks (apply indicators & generate signals)
    processed_data = process_stocks(df, stocks)

    # Step 3: Prepare data and train ML model
    all_data = pd.concat(processed_data.values(), keys=processed_data.keys())
    model, X = train_ml_model(all_data)

    # Step 4: Predict trades
    trade_signals = predict_trades(model, X, stocks, processed_data)

    # Step 5: Display results
    for stock, decision in trade_signals.items():
        print(decision)




# Run the script
if __name__ == "__main__":
    main()
