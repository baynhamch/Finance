import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# 1) Get OHLCV data
df = yf.download("BTC-USD", start="2023-01-01", end="2024-01-01", interval="1d")

# 2) Build features
df["return"] = df["Close"].pct_change()
df["log_volume"] = np.log(df["Volume"] + 1)

# Drop missing rows from first diff
df = df.dropna(subset=["return", "log_volume"])

# 3) Pearson & Spearman (single-number, whole period)
pearson_r, pearson_p = pearsonr(df["log_volume"], df["return"])
spearman_rho, spearman_p = spearmanr(df["log_volume"], df["return"])

print(f"Pearson r = {pearson_r:.4f} (p={pearson_p:.3g})")
print(f"Spearman ρ = {spearman_rho:.4f} (p={spearman_p:.3g})")

# 4) The same via pandas (no p-values)
print("pandas Pearson:", df["log_volume"].corr(df["return"], method="pearson"))
print("pandas Spearman:", df["log_volume"].corr(df["return"], method="spearman"))

# 5) Rolling correlations (see regime changes over time)
win = 30  # 30 trading days ≈ 1–1.5 months
roll_pearson = df["log_volume"].rolling(win).corr(df["return"])
# For Spearman rolling, rank inside each window:
def rolling_spearman(x, y, window):
    # Rank within each window, then compute Pearson on ranks (equivalent to Spearman)
    # We'll use a loop for clarity; for production, vectorize or use numba.
    out = pd.Series(index=x.index, dtype=float)
    for i in range(window-1, len(x)):
        xx = x.iloc[i-window+1:i+1].rank()
        yy = y.iloc[i-window+1:i+1].rank()
        out.iloc[i] = xx.corr(yy, method="pearson")
    return out

roll_spearman = rolling_spearman(df["log_volume"], df["return"], win)

# 6) Quick visuals
plt.figure()
plt.scatter(df["log_volume"], df["return"], alpha=0.5)
plt.xlabel("log(volume)")
plt.ylabel("daily return")
plt.title("BTC: Volume vs Return (scatter)")
plt.show()

plt.figure()
roll_pearson.plot(label="Rolling Pearson", linewidth=1.5)
roll_spearman.plot(label="Rolling Spearman", linewidth=1.5)
plt.axhline(0, linestyle="--")
plt.legend()
plt.title(f"BTC: {win}-Day Rolling Correlations (Volume vs Return)")
plt.ylabel("correlation")
plt.show()