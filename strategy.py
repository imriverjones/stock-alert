"""Momentum swing-trade rules, shared by the backtest and the live scanner.

A stock gives an ENTRY signal on a day when ALL of these are true:
  1. Trend:     close > 50-day average > 200-day average
  2. Strength:  its 63-day (about 3-month) return beats its market benchmark
  3. RSI(14):   between 50 and 70
  4. Breakout:  close is above the highest close of the previous 20 days
  5. Volume:    volume is at least 1.5x its previous 20-day average
plus a liquidity filter: average daily traded value over the previous
20 days of at least $1m (US) or GBP 250k (UK).

Suggested stop:  signal-day close minus 2 x ATR(14).
Exit signal:     a close below the 20-day average.
Position size:   risk no more than 1% of the account if the stop is hit.

These rules find candidates; they do not predict that a trade will win.
"""

import numpy as np
import pandas as pd

SETTINGS = {
    "sma_fast": 50,
    "sma_slow": 200,
    "sma_exit": 20,
    "rs_days": 63,
    "rsi_days": 14,
    "rsi_min": 50,
    "rsi_max": 70,
    "breakout_days": 20,
    "volume_days": 20,
    "volume_multiple": 1.5,
    "atr_days": 14,
    "atr_stop_multiple": 2.0,
    "min_traded_value": {"US": 1_000_000, "UK": 250_000},
    "risk_per_trade": 0.01,
}

BENCHMARKS = {"US": "SPY", "UK": "ISF.L"}


def price_divisor(currency):
    """UK prices from Yahoo/Trading 212 are usually in pence (GBX)."""
    return 100.0 if currency == "GBX" else 1.0


def clean_prices(df):
    """Drop rows with missing or zero prices and one-day data glitches.

    A glitch is a close more than 4x (or less than a quarter of) BOTH the
    day before and the day after, which is almost always a bad data point
    (for example a price reported in pounds instead of pence).
    """
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df["Close"] > 0) & (df["Open"] > 0)]
    if len(df) < 3:
        return df
    close = df["Close"]
    prev_ratio = close / close.shift(1)
    next_ratio = close / close.shift(-1)
    spike = ((prev_ratio > 4) & (next_ratio > 4)) | ((prev_ratio < 0.25) & (next_ratio < 0.25))
    return df[~spike.fillna(False)]


def _wilder_average(values, days):
    """Wilder's smoothing: simple average of the first `days` values, then
    each new value = (previous * (days - 1) + current) / days."""
    out = np.full(len(values), np.nan)
    start = None
    for k in range(len(values)):
        if np.isnan(values[k]):
            continue
        if start is None:
            start = k
        if k - start + 1 == days:
            out[k] = np.mean(values[start:k + 1])
        elif k - start + 1 > days:
            out[k] = (out[k - 1] * (days - 1) + values[k]) / days
    return out


def rsi(close, days):
    """Standard (Wilder) RSI."""
    change = close.diff().to_numpy()
    gain = np.where(np.isnan(change), np.nan, np.clip(change, 0, None))
    loss = np.where(np.isnan(change), np.nan, np.clip(-change, 0, None))
    avg_gain = _wilder_average(gain, days)
    avg_loss = _wilder_average(loss, days)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(avg_loss == 0, 100.0, 100 - 100 / (1 + avg_gain / avg_loss))
    value = np.where(np.isnan(avg_gain), np.nan, value)
    return pd.Series(value, index=close.index)


def atr(df, days):
    prev_close = df["Close"].shift(1)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / days, adjust=False, min_periods=days).mean()


def add_indicators(df, benchmark_close, market, currency):
    """Return a copy of an OHLCV DataFrame with indicator and signal columns.

    Every value on a row uses only that day's and earlier data.
    """
    s = SETTINGS
    df = df.copy()
    close = df["Close"]

    df["sma_fast"] = close.rolling(s["sma_fast"]).mean()
    df["sma_slow"] = close.rolling(s["sma_slow"]).mean()
    df["sma_exit"] = close.rolling(s["sma_exit"]).mean()
    df["rsi"] = rsi(close, s["rsi_days"])
    df["atr"] = atr(df, s["atr_days"])

    df["ret_63"] = close / close.shift(s["rs_days"]) - 1
    bench = benchmark_close.reindex(df.index).ffill()
    df["bench_ret_63"] = bench / bench.shift(s["rs_days"]) - 1

    df["prior_high_close"] = close.shift(1).rolling(s["breakout_days"]).max()
    df["avg_volume"] = df["Volume"].shift(1).rolling(s["volume_days"]).mean()
    df["volume_ratio"] = df["Volume"] / df["avg_volume"].replace(0, np.nan)

    traded_value = close * df["Volume"] / price_divisor(currency)
    df["avg_traded_value"] = traded_value.shift(1).rolling(s["volume_days"]).mean()

    df["rule_liquid"] = df["avg_traded_value"] >= s["min_traded_value"][market]
    df["rule_trend"] = (close > df["sma_fast"]) & (df["sma_fast"] > df["sma_slow"])
    df["rule_strength"] = df["ret_63"] > df["bench_ret_63"]
    df["rule_rsi"] = df["rsi"].between(s["rsi_min"], s["rsi_max"])
    df["rule_breakout"] = close > df["prior_high_close"]
    df["rule_volume"] = df["volume_ratio"] >= s["volume_multiple"]

    rules = ["rule_liquid", "rule_trend", "rule_strength", "rule_rsi",
             "rule_breakout", "rule_volume"]
    df["signal"] = df[rules].fillna(False).all(axis=1)
    df["stop"] = close - s["atr_stop_multiple"] * df["atr"]
    return df


def explain(row, market, currency):
    """Plain-English reasons for a signal row, for alerts and logs."""
    if currency == "GBX":
        price = f"{row['Close']:.2f}p"
    elif currency == "USD":
        price = f"${row['Close']:.2f}"
    else:
        price = f"{row['Close']:.2f} {currency}"
    return [
        f"Uptrend: close {price} > 50d avg {row['sma_fast']:.2f} > 200d avg {row['sma_slow']:.2f}",
        f"Beating market: 3-month return {row['ret_63']:+.1%} vs benchmark {row['bench_ret_63']:+.1%}",
        f"RSI(14): {row['rsi']:.0f}",
        f"Breakout: above previous 20-day high close of {row['prior_high_close']:.2f}",
        f"Volume: {row['volume_ratio']:.1f}x its 20-day average",
        f"Suggested stop: {row['stop']:.2f} (2 x ATR {row['atr']:.2f} below close)",
    ]
