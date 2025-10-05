import pandas as pd
import numpy as np

# Moving avergae
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

# Releative Stength Index
def rsi(series: pd.Series, n: int=14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

"""
True Range 
•	Measures how much the market actually moved in one bar.
•	Combines gap moves and intrabar range:
	1.	Current high–low
	2.	High–previous close
	3.	Low–previous close
•	Takes the largest of the three = “true” range.
Used for volatility and stop sizing.
"""
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

"""
Average True Range
	•	Smooths True Range over n bars (default 14).
	•	Gives average volatility.
	•	Traders use it to size positions or set stops:
	•	“SL = 1×ATR below entry”
	•	“TP = 0.8×ATR above entry.”
"""
def atr(df: pd.DataFrame, n: int=14) -> pd.Series:
    return true_range(df).rolling(n).mean()

"""
High = n
	•	Returns the highest high over the last n bars.
	•	In this strategy, used to confirm a 2-bar breakout (“close breaks a 2-bar high → momentum confirmation”).
"""
def high_n(df: pd.DataFrame, n: int=2) -> pd.Series:
    return df['High'].rolling(n).max()
