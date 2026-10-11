"""Backtest the momentum rules in strategy.py on about 5 years of daily data.

How each simulated trade works:
  - Signal at the close of day D (all rules true).
  - Buy at the open of day D+1.
  - Sell if the price falls to the stop (or at the open if it gaps below).
  - Otherwise sell at the next open after a close below the 20-day average.
  - Costs are deducted from every trade (see COSTS below).

Results are reported in "R": 1R is the planned risk: signal-day close minus stop (2 x ATR).
A trade that makes +2R made twice what was risked; -1R means the stop hit.

Writes results/summary.md and results/trades.csv.
"""

import os
import sys
import time
from collections import defaultdict

import pandas as pd
import yfinance as yf

import strategy
from universe import load_universe

PERIOD = "5y"
BATCH_SIZE = 100

# Round-trip trading costs as a fraction of the entry price (rough estimates).
COSTS = {
    "US": 0.004,        # Trading 212 FX fee 0.15% each way plus slippage
    "UK": 0.008,        # 0.5% stamp duty plus slippage
    "UK_AIM": 0.003,    # AIM shares are exempt from stamp duty
    "CHEAP_EXTRA": 0.015,  # wider buy/sell spreads on shares under $1 / 100p
}


def download_batch(tickers):
    for attempt in range(1, 4):
        try:
            data = yf.download(
                tickers, period=PERIOD, interval="1d", group_by="ticker",
                auto_adjust=True, threads=True, progress=False,
            )
            return data
        except Exception as error:
            print(f"  download error (attempt {attempt}): {error}")
            time.sleep(10 * attempt)
    return None


def split_batch(data, tickers):
    frames = {}
    if data is None or data.empty:
        return frames
    if not isinstance(data.columns, pd.MultiIndex):
        frames[tickers[0]] = data
        return frames
    available = set(data.columns.get_level_values(0))
    for ticker in tickers:
        if ticker in available:
            frames[ticker] = data[ticker]
    return frames


def simulate(df, info):
    """Return a list of trade dicts for one stock."""
    trades = []
    divisor = strategy.price_divisor(info["currency"])
    opens = df["Open"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    sma_exit = df["sma_exit"].to_numpy()
    stops = df["stop"].to_numpy()
    signals = df["signal"].to_numpy()
    dates = df.index
    n = len(df)

    i = 0
    while i < n - 1:
        if not signals[i]:
            i += 1
            continue

        entry = opens[i + 1]
        stop = stops[i]
        if not (entry > stop > 0):
            i += 1
            continue

        exit_price, exit_index, reason = None, None, None
        j = i + 1
        while j < n:
            if lows[j] <= stop:
                exit_price = stop if j == i + 1 else min(stop, opens[j])
                exit_index, reason = j, "stop"
                break
            if closes[j] < sma_exit[j]:
                if j + 1 < n:
                    exit_price, exit_index, reason = opens[j + 1], j + 1, "trend exit"
                else:
                    exit_price, exit_index, reason = closes[j], j, "still open"
                break
            j += 1
        if exit_price is None:
            exit_price, exit_index, reason = closes[-1], n - 1, "still open"

        market_key = "UK_AIM" if info["aim"] else info["market"]
        cheap = closes[i] / divisor < 1
        cost = COSTS[market_key] + (COSTS["CHEAP_EXTRA"] if cheap else 0)
        # 1R is the planned risk at the signal: signal-day close minus stop
        # (2 x ATR), so overnight gaps do not distort R.
        risk = closes[i] - stop
        gross = exit_price - entry
        net = gross - cost * entry

        trades.append({
            "ticker": info["yahoo"],
            "name": info["name"],
            "market": info["market"],
            "aim": info["aim"],
            "price_band": "under 1" if cheap else "1 and over",
            "signal_date": dates[i].date(),
            "entry_date": dates[i + 1].date(),
            "exit_date": dates[exit_index].date(),
            "days_held": exit_index - (i + 1),
            "entry": round(entry, 4),
            "stop": round(stop, 4),
            "exit": round(exit_price, 4),
            "exit_reason": reason,
            "return_pct": round(net / entry * 100, 2),
            "r_multiple": round(net / risk, 3),
            "gross_r": round(gross / risk, 3),
        })
        i = exit_index + 1
    return trades


def stats(trades):
    if trades.empty:
        return None
    wins = trades[trades["r_multiple"] > 0]
    losses = trades[trades["r_multiple"] <= 0]
    loss_total = -losses["r_multiple"].sum()
    return {
        "Trades": len(trades),
        "Win rate": f"{len(wins) / len(trades):.0%}",
        "Avg R (after costs)": f"{trades['r_multiple'].mean():+.2f}",
        "Avg R (before costs)": f"{trades['gross_r'].mean():+.2f}",
        "Avg return": f"{trades['return_pct'].mean():+.1f}%",
        "Avg win": f"{wins['return_pct'].mean():+.1f}%" if len(wins) else "-",
        "Avg loss": f"{losses['return_pct'].mean():+.1f}%" if len(losses) else "-",
        "Profit factor": f"{wins['r_multiple'].sum() / loss_total:.2f}" if loss_total > 0 else "-",
        "Median days held": int(trades["days_held"].median()),
    }


def table(groups):
    rows = [(name, stats(t)) for name, t in groups]
    rows = [(name, s) for name, s in rows if s]
    if not rows:
        return "_No trades._\n"
    headers = ["Group"] + list(rows[0][1].keys())
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for name, s in rows:
        lines.append("| " + " | ".join([name] + [str(v) for v in s.values()]) + " |")
    return "\n".join(lines) + "\n"


def main():
    universe = load_universe()
    by_ticker = {u["yahoo"]: u for u in universe}

    print("Downloading benchmarks...")
    benchmarks = {}
    for market, symbol in strategy.BENCHMARKS.items():
        data = yf.download(symbol, period=PERIOD, interval="1d",
                           auto_adjust=True, progress=False)
        if data is None or data.empty:
            print(f"ERROR: could not download benchmark {symbol}.")
            sys.exit(1)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        benchmarks[market] = close
        print(f"  {symbol}: {len(close)} days")

    tickers = list(by_ticker)
    all_trades = []
    signals_per_day = defaultdict(lambda: defaultdict(int))
    downloaded = 0
    too_short = 0

    for start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[start:start + BATCH_SIZE]
        print(f"Batch {start // BATCH_SIZE + 1} of {-(-len(tickers) // BATCH_SIZE)}: "
              f"{len(batch)} tickers")
        frames = split_batch(download_batch(batch), batch)
        for ticker, df in frames.items():
            info = by_ticker[ticker]
            df = strategy.clean_prices(df)
            if len(df) < strategy.SETTINGS["sma_slow"] + 30:
                too_short += 1
                continue
            downloaded += 1
            df = strategy.add_indicators(df, benchmarks[info["market"]],
                                         info["market"], info["currency"])
            for day in df.index[df["signal"]]:
                signals_per_day[info["market"]][day.date()] += 1
            all_trades.extend(simulate(df, info))
        time.sleep(1)

    failed = len(tickers) - downloaded - too_short
    print(f"Price history used for {downloaded} stocks; {too_short} had under "
          f"{strategy.SETTINGS['sma_slow'] + 30} days of data; {failed} returned no data.")

    if downloaded == 0:
        print("ERROR: no usable price data. Yahoo may be blocking requests; try again later.")
        sys.exit(1)

    trades = pd.DataFrame(all_trades)
    os.makedirs("results", exist_ok=True)
    trades.to_csv("results/trades.csv", index=False)

    closed = trades[trades["exit_reason"] != "still open"] if not trades.empty else trades
    lines = ["# Momentum strategy backtest", ""]
    lines.append(f"Stocks tested: {downloaded} (period: last {PERIOD}, daily bars). "
                 f"Closed trades: {len(closed)}.")
    lines.append("")
    lines.append("1R = the planned risk per trade (signal-day close minus stop, i.e. 2 x ATR). "
                 "Avg R above zero after costs is the minimum for an edge.")
    lines.append("")

    if closed.empty:
        lines.append("_No closed trades - the rules never triggered._")
    else:
        lines.append("## Overall")
        lines.append(table([("All trades", closed)]))
        lines.append("## By market")
        lines.append(table([
            ("US", closed[closed["market"] == "US"]),
            ("UK main market", closed[(closed["market"] == "UK") & (~closed["aim"])]),
            ("UK AIM", closed[(closed["market"] == "UK") & (closed["aim"])]),
        ]))
        lines.append("## By share price (cheap vs normal)")
        lines.append(table([
            ("Under $1 / 100p", closed[closed["price_band"] == "under 1"]),
            ("$1 / 100p and over", closed[closed["price_band"] == "1 and over"]),
        ]))
        years = sorted(pd.to_datetime(closed["entry_date"]).dt.year.unique())
        lines.append("## By year")
        lines.append(table([
            (str(y), closed[pd.to_datetime(closed["entry_date"]).dt.year == y]) for y in years
        ]))

    lines.append("## How many alerts would you get?")
    for market in ("US", "UK"):
        counts = signals_per_day.get(market, {})
        total_days = len(benchmarks[market]) - strategy.SETTINGS["sma_slow"]
        total = sum(counts.values())
        busiest = max(counts.values()) if counts else 0
        avg = total / total_days if total_days > 0 else 0
        lines.append(f"- {market}: about {avg:.2f} signals per trading day on average; "
                     f"busiest day had {busiest}.")
    lines.append("")
    lines.append("## Important limitations")
    lines.append("- Only stocks listed today are tested. Companies that went bust or were "
                 "delisted are missing, which makes results look better than reality - "
                 "especially for cheap and small stocks.")
    lines.append("- Yahoo data can contain errors; costs and spreads are rough estimates.")
    lines.append("- Each trade is measured on its own; it ignores how many you could afford "
                 "to hold at once.")
    lines.append("- Past results do not guarantee future results.")

    summary = "\n".join(lines) + "\n"
    with open("results/summary.md", "w", encoding="utf-8") as handle:
        handle.write(summary)
    print("")
    print(summary)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(summary)


if __name__ == "__main__":
    main()
