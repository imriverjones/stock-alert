"""Build the list of stocks to scan from Trading 212's instrument list.

Keeps only ordinary shares on NASDAQ, NYSE, the London Stock Exchange
and London AIM, and converts Trading 212 tickers into Yahoo Finance
tickers (for example AAPL_US_EQ -> AAPL, BARCl_EQ -> BARC.L).
"""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "https://live.trading212.com/api/v0/equity/metadata"

# Trading 212 exchange name -> (market, is AIM)
EXCHANGES = {
    "NASDAQ": ("US", False),
    "NYSE": ("US", False),
    "London Stock Exchange": ("UK", False),
    "London Stock Exchange AIM": ("UK", True),
}


def _get_json(path, headers):
    for attempt in range(1, 4):
        request = urllib.request.Request(f"{BASE_URL}/{path}", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < 3:
                print(f"Trading 212 rate limit on /{path}; waiting 55 seconds.")
                time.sleep(55)
                continue
            hints = {
                401: "the API key or secret is wrong.",
                403: "the key is missing the Metadata permission.",
            }
            print(f"ERROR: Trading 212 returned HTTP {error.code} for /{path}: "
                  f"{hints.get(error.code, 'unexpected error.')}")
            sys.exit(1)
    print(f"ERROR: still rate limited on /{path} after 3 attempts.")
    sys.exit(1)


def to_yahoo(ticker, market):
    """Convert a Trading 212 ticker to a Yahoo Finance ticker, or None."""
    if market == "US":
        if not ticker.endswith("_US_EQ"):
            return None
        symbol = ticker[: -len("_US_EQ")]
        return symbol.replace(".", "-").replace("_", "-").replace("/", "-")
    if market == "UK":
        if not ticker.endswith("l_EQ"):
            return None
        symbol = ticker[: -len("l_EQ")]
        symbol = symbol.replace(".", "-").replace("_", "-").replace("/", "-")
        return symbol + ".L"
    return None


def load_universe():
    """Return a list of dicts: yahoo, t212, name, market, aim, currency."""
    key = os.environ.get("TRADING212_API_KEY", "").strip()
    secret = os.environ.get("TRADING212_API_SECRET", "").strip()
    if not key or not secret:
        print("ERROR: TRADING212_API_KEY or TRADING212_API_SECRET is missing or empty.")
        sys.exit(1)

    credentials = base64.b64encode(f"{key}:{secret}".encode("utf-8")).decode("ascii")
    headers = {"Authorization": f"Basic {credentials}"}

    instruments = _get_json("instruments", headers)
    exchanges = _get_json("exchanges", headers)

    schedule_to_exchange = {}
    for exchange in exchanges:
        for schedule in exchange.get("workingSchedules", []):
            schedule_to_exchange[schedule.get("id")] = exchange.get("name")

    universe = []
    unmapped = 0
    seen = set()
    for item in instruments:
        if item.get("type") != "STOCK":
            continue
        exchange_name = schedule_to_exchange.get(item.get("workingScheduleId"))
        if exchange_name not in EXCHANGES:
            continue
        market, is_aim = EXCHANGES[exchange_name]
        yahoo = to_yahoo(item.get("ticker", ""), market)
        if not yahoo:
            unmapped += 1
            continue
        if yahoo in seen:
            continue
        seen.add(yahoo)
        universe.append({
            "yahoo": yahoo,
            "t212": item.get("ticker"),
            "name": item.get("name", ""),
            "market": market,
            "aim": is_aim,
            "currency": item.get("currencyCode", ""),
        })

    us = sum(1 for u in universe if u["market"] == "US")
    uk = len(universe) - us
    print(f"Universe: {len(universe)} stocks ({us} US, {uk} UK); "
          f"{unmapped} skipped because the ticker format was not recognised.")
    if not universe:
        print("ERROR: universe is empty.")
        sys.exit(1)
    return universe
