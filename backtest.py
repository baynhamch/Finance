"""
Simple Backtesting Script using Moving Average Crossover Strategy

This script downloads historical stock data using yfinance, applies a 
simple moving average (SMA) crossover strategy, and compares the 
strategy’s performance to a buy-and-hold approach.

Author: [Your Name]
Date: [Date]
"""

import pandas as pd
import yfinance as yf

def backtest(symbol="AAPL", start="2022-01-01", end="2023-01-01"):
    """
    Backtest a simple moving average crossover strategy.

    Parameters
    ----------
    symbol : str
        Ticker symbol of the stock (default: "AAPL").
    start : str
        Start date for historical data in "YYYY-MM-DD" format.
    end : str
        End date for historical data in "YYYY-MM-DD" format.

    Returns
    -------
    pd.DataFrame
        DataFrame containing price data, moving averages, signals,
        and calculated returns.
    """

    # 1. Load historical daily data from Yahoo Finance
    data = yf.download(symbol, start=start, end=end)
    data = data[["Close"]].dropna()  # Keep only the closing price

    # 2. Generate strategy signals using SMA crossover
    data["SMA_short"] = data["Close"].rolling(window=20).mean()  # Short-term average (20 days)
    data["SMA_long"] = data["Close"].rolling(window=50).mean()   # Long-term average (50 days)

    # Initialize signals column: 1 = Buy, -1 = Sell, 0 = No position
    data["Signal"] = 0
    data.loc[data["SMA_short"] > data["SMA_long"], "Signal"] = 1   # Bullish crossover → Buy
    data.loc[data["SMA_short"] < data["SMA_long"], "Signal"] = -1  # Bearish crossover → Sell

    # 3. Calculate returns
    data["Market_Return"] = data["Close"].pct_change()  # Daily returns of the stock
    # Strategy return = previous day's signal × today's market return
    data["Strategy_Return"] = data["Signal"].shift(1) * data["Market_Return"]

    # 4. Compute overall performance
    total_return = (data["Strategy_Return"] + 1).prod() - 1  # Compound strategy returns
    buy_hold_return = (data["Market_Return"] + 1).prod() - 1 # Buy & hold return

    # Print results
    print(f"Symbol: {symbol}")
    print(f"Backtest period: {start} → {end}")
    print(f"Strategy return: {total_return:.2%}")
    print(f"Buy & Hold return: {buy_hold_return:.2%}")

    return data


if __name__ == "__main__":
    # Example: Backtest Apple stock in 2022
    df = backtest("AAPL", "2022-01-01", "2023-01-01")

    # Display the last few rows with signals and returns
    print(df.tail())
