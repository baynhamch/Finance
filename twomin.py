# twomin.py
#  https://www.wealth-lab.com/Strategy/DesignPublished?strategyID=10
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --------- helpers: consecutive-down logic ----------
def consec_run(series: pd.Series, direction: str = "down") -> pd.Series:
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    run = np.zeros(n, dtype=int)
    for i in range(1, n):
        if direction == "down":
            run[i] = run[i-1] + 1 if vals[i] < vals[i-1] else 0
        else:
            run[i] = run[i-1] + 1 if vals[i] > vals[i-1] else 0
    return pd.Series(run, index=series.index)

def consec_down(series: pd.Series, n: int = 3) -> pd.Series:
    return consec_run(series, "down").eq(n)

def make_profit_target(entry_px: float, target_pct: float) -> float:
    return entry_px * (1 + target_pct)

# --------- main backtest (fixed n=3) ----------
def backtest_2minute(
    symbol="SPY",
    start="2024-01-01",
    end="2025-01-01",
    interval="1d",
    target_pct=0.01,
    use_time_exit=True,
    max_hold=5,
    use_atr_stop=False,
    atr_len=14,
    atr_mult=2.0,
    initial_cash=100_000,
    stake_dollars=10_000,
    fee_bps=0.0,
    slip_bps=0.0
):
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True).dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, [c for c in df.columns if c in ['Open','High','Low','Close','Volume']]].copy()
    df = df.loc[:, ~df.columns.duplicated()]
    for c in ['Open','High','Low','Close','Volume']:
        col = df[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        df[c] = pd.to_numeric(col, errors='coerce')
    df.dropna(inplace=True)
    if df.empty or len(df) < 10:
        raise RuntimeError("No data returned")

    signal = consec_down(df['Close'], 3)

    if use_atr_stop:
        tr1 = (df['High'] - df['Low']).abs()
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(atr_len, min_periods=atr_len).mean()

    fee = fee_bps / 10_000.0
    slip = slip_bps / 10_000.0
    cash = initial_cash
    pos = None
    eq, trades = [], []

    for i in range(len(df) - 1):
        close_i = df['Close'].iloc[i]
        eq.append(cash + (pos['qty']*close_i if pos else 0.0))

        if (pos is None) and bool(signal.iloc[i]):
            entry_open = df['Open'].iloc[i+1] * (1 + slip)
            qty = int(stake_dollars // entry_open)
            if qty > 0:
                fee_entry = entry_open * qty * fee
                cash -= entry_open * qty + fee_entry
                target_px = make_profit_target(entry_open, target_pct)
                stop_px = None
                if use_atr_stop and ('ATR' in df.columns) and not np.isnan(df['ATR'].iloc[i+1]):
                    stop_px = entry_open - df['ATR'].iloc[i+1] * atr_mult
                pos = {'entry_idx': i+1, 'entry_px': entry_open, 'qty': qty,
                       'target_px': target_px, 'stop_px': stop_px, 'fees': fee_entry}
                hi, lo = df['High'].iloc[i+1], df['Low'].iloc[i+1]
                if (stop_px is not None and lo <= stop_px) or (hi >= target_px):
                    if stop_px is not None and lo <= stop_px:
                        exit_px, reason = stop_px * (1 - slip), 'stop_same_bar'
                    else:
                        exit_px, reason = target_px * (1 - slip), 'target_same_bar'
                    fee_exit = exit_px * qty * fee
                    cash += exit_px * qty - fee_exit
                    pnl = (exit_px - entry_open) * qty - fee_entry - fee_exit
                    trades.append({'entry_time': df.index[i+1],'exit_time': df.index[i+1],
                                   'entry_px': entry_open,'exit_px': exit_px,'qty': qty,
                                   'pnl': pnl,'bars_held': 0,'reason': reason})
                    pos = None

        elif pos is not None:
            hi, lo = df['High'].iloc[i], df['Low'].iloc[i]
            exit_px, reason = None, None
            if pos['stop_px'] is not None and lo <= pos['stop_px']:
                exit_px, reason = pos['stop_px'] * (1 - slip), 'stop'
            elif hi >= pos['target_px']:
                exit_px, reason = pos['target_px'] * (1 - slip), 'target'
            if (exit_px is None) and use_time_exit and (i - pos['entry_idx'] + 1 >= max_hold):
                exit_px, reason = df['Close'].iloc[i], 'time_exit'
            if exit_px is not None:
                qty = pos['qty']
                fee_exit = exit_px * qty * fee
                cash += exit_px * qty - fee_exit
                pnl = (exit_px - pos['entry_px']) * qty - pos['fees'] - fee_exit
                trades.append({'entry_time': df.index[pos['entry_idx']], 'exit_time': df.index[i],
                               'entry_px': pos['entry_px'], 'exit_px': exit_px, 'qty': qty,
                               'pnl': pnl, 'bars_held': i - pos['entry_idx'], 'reason': reason})
                pos = None

    last_close = df['Close'].iloc[-1]
    eq.append(cash + (pos['qty']*last_close if pos else 0.0))
    equity = pd.Series(eq, index=df.index[:len(eq)], name='Equity')
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        winrate = (trades_df['pnl'] > 0).mean()
        gp = trades_df.loc[trades_df.pnl > 0, 'pnl'].sum()
        gl = -trades_df.loc[trades_df.pnl < 0, 'pnl'].sum()
        pf = gp / gl if gl > 0 else np.inf
    else:
        winrate, pf = np.nan, np.nan
    print(f"{symbol} | Trades: {len(trades_df)} | Win%: {winrate:.1%} | PF: {pf:.2f} | Final Equity: {equity.iloc[-1]:.2f}")
    return df, trades_df, equity

# --------- grid-search version ----------
def backtest_2minute_with_n(symbol="SPY", start="2024-01-01", end="2025-01-01",
                             interval="1d", target_pct=0.01, n_down=3,
                             use_time_exit=True, max_hold=5,
                             initial_cash=100_000, stake_dollars=10_000,
                             fee_bps=1.0, slip_bps=0.5):
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True).dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, [c for c in df.columns if c in ['Open','High','Low','Close','Volume']]].copy()
    df = df.loc[:, ~df.columns.duplicated()]
    for c in ['Open','High','Low','Close','Volume']:
        col = df[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        df[c] = pd.to_numeric(col, errors='coerce')
    df.dropna(inplace=True)
    if df.empty or len(df) < 10:
        return None, None, None

    def _consec_down(series, n=3):
        vals = series.to_numpy(dtype=float)
        run = np.zeros(len(vals), dtype=int)
        for i in range(1, len(vals)):
            run[i] = run[i-1] + 1 if vals[i] < vals[i-1] else 0
        return pd.Series(run, index=series.index).eq(n)

    signal = _consec_down(df['Close'], n_down)
    fee = fee_bps / 10_000.0
    slip = slip_bps / 10_000.0
    cash, pos, eq, trades = initial_cash, None, [], []

    for i in range(len(df) - 1):
        close_i = df['Close'].iloc[i]
        eq.append(cash + (pos['qty']*close_i if pos else 0.0))
        if (pos is None) and bool(signal.iloc[i]):
            entry_open = df['Open'].iloc[i+1] * (1 + slip)
            qty = int(stake_dollars // entry_open)
            if qty > 0:
                fee_entry = entry_open * qty * fee
                cash -= entry_open * qty + fee_entry
                tgt = entry_open * (1 + target_pct)
                pos = {'entry_idx': i+1, 'entry_px': entry_open, 'qty': qty, 'tgt': tgt, 'fees': fee_entry}
                hi = df['High'].iloc[i+1]
                if hi >= tgt:
                    exit_px = tgt * (1 - slip)
                    fee_exit = exit_px * qty * fee
                    cash += exit_px * qty - fee_exit
                    pnl = (exit_px - entry_open) * qty - fee_entry - fee_exit
                    trades.append({'entry_time': df.index[i+1], 'exit_time': df.index[i+1],
                                   'entry_px': entry_open, 'exit_px': exit_px, 'qty': qty,
                                   'pnl': pnl, 'bars_held': 0, 'reason':'target_same_bar'})
                    pos = None
        elif pos is not None:
            hi = df['High'].iloc[i]
            exit_px, reason = None, None
            if hi >= pos['tgt']:
                exit_px, reason = pos['tgt'] * (1 - slip), 'target'
            if (exit_px is None) and use_time_exit and (i - pos['entry_idx'] + 1 >= max_hold):
                exit_px, reason = df['Close'].iloc[i], 'time_exit'
            if exit_px is not None:
                qty = pos['qty']
                fee_exit = exit_px * qty * fee
                cash += exit_px * qty - fee_exit
                pnl = (exit_px - pos['entry_px']) * qty - pos['fees'] - fee_exit
                trades.append({'entry_time': df.index[pos['entry_idx']], 'exit_time': df.index[i],
                               'entry_px': pos['entry_px'], 'exit_px': exit_px, 'qty': qty,
                               'pnl': pnl, 'bars_held': i - pos['entry_idx'], 'reason': reason})
                pos = None

    last_close = df['Close'].iloc[-1]
    eq.append(cash + (pos['qty']*last_close if pos else 0.0))
    equity = pd.Series(eq, index=df.index[:len(eq)], name='Equity')
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return df, trades_df, equity
    winrate = (trades_df['pnl'] > 0).mean()
    gp = trades_df.loc[trades_df.pnl > 0, 'pnl'].sum()
    gl = -trades_df.loc[trades_df.pnl < 0, 'pnl'].sum()
    pf = gp / gl if gl > 0 else np.inf
    print(f"{symbol} n={n_down} tgt={target_pct*100:.1f}% | Trades:{len(trades_df)} Win%:{winrate:.1%} PF:{pf:.2f} Final:{equity.iloc[-1]:.2f}")
    return df, trades_df, equity

# --------- visualization function ---------
def plot_strategy(df, trades_df, equity, symbol="SPY", n_down=3, target_pct=0.01):
    """Plot price with entry/exit markers and equity curve"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Top panel: Price chart with trades
    ax1.plot(df.index, df['Close'], 'k-', linewidth=1, label='Close Price', alpha=0.7)

    if not trades_df.empty:
        # Entry points
        entries = trades_df[['entry_time', 'entry_px']].drop_duplicates()
        ax1.scatter(entries['entry_time'], entries['entry_px'],
                   marker='^', color='green', s=100, label='Entry', zorder=5)

        # Exit points - color by profit/loss
        for _, trade in trades_df.iterrows():
            color = 'lime' if trade['pnl'] > 0 else 'red'
            ax1.scatter(trade['exit_time'], trade['exit_px'],
                       marker='v', color=color, s=100, zorder=5)

        # Draw lines connecting entries to exits
        for _, trade in trades_df.iterrows():
            color = 'green' if trade['pnl'] > 0 else 'red'
            ax1.plot([trade['entry_time'], trade['exit_time']],
                    [trade['entry_px'], trade['exit_px']],
                    color=color, alpha=0.3, linewidth=1)

    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_title(f'{symbol} - ConsecDown({n_down}) Strategy (Target: {target_pct*100:.1f}%)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Equity curve with profit bars
    ax2.plot(equity.index, equity.values, 'b-', linewidth=2, label='Equity Curve')
    ax2.axhline(y=equity.iloc[0], color='gray', linestyle='--', alpha=0.5, label='Starting Capital')

    # Add profit/loss bars at trade exit times
    if not trades_df.empty:
        for _, trade in trades_df.iterrows():
            color = 'green' if trade['pnl'] > 0 else 'red'
            ax2.axvline(x=trade['exit_time'], color=color, alpha=0.2, linewidth=0.5)

    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Equity ($)', fontsize=11)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# --------- main runner ---------
if __name__ == "__main__":
    results = []
    for n_down in (2,3,4):
        for tgt in (0.005, 0.008, 0.010, 0.012, 0.015):
            df, trades, eq = backtest_2minute_with_n(
                symbol="AAPL",
                start="2020-01-01", end="2025-09-01",
                interval="1d",
                target_pct=tgt,
                n_down=n_down,
                use_time_exit=True, max_hold=5,
                stake_dollars=10_000,
                fee_bps=1.0,
                slip_bps=0.5
            )
            if trades is not None and not trades.empty:
                results.append({'n_down': n_down, 'tgt': tgt, 'df': df, 'trades': trades, 'eq': eq})

    # Plot all equity curves on one graph
    if results:
        plt.figure(figsize=(14, 8))
        for r in results:
            label = f"n={r['n_down']}, tgt={r['tgt']*100:.1f}%"
            plt.plot(r['eq'].index, r['eq'].values, linewidth=1.5, label=label, alpha=0.7)

        plt.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
        plt.xlabel('Date', fontsize=11)
        plt.ylabel('Equity ($)', fontsize=11)
        plt.title('AAPL - All Strategy Variations (ConsecDown)', fontsize=13, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
