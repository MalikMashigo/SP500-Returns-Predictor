"""
news_fetcher.py
---------------
Fetches financial news headlines from the Alpaca News API and
preprocesses them for use by the FinBERT encoder.

Known issues (Part 3 interim):
  - ~80-100 training days have zero headlines (imputed with zero vectors).
  - Variable number of headlines per day leads to inconsistent pooled embeddings.
  - Currently using mean pooling; top-k and attention pooling under investigation.

Alpaca News API docs:
  https://alpaca.markets/docs/api-references/market-data-api/news/

Requirements:
    pip install alpaca-py requests
"""

import os
import json
import time
from datetime import date, timedelta
from typing import Optional

import requests
import pandas as pd


ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET  = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

MARKET_SYMBOLS = ["SPY", "^GSPC", "QQQ"]  # broad market coverage
TOP_K_HEADLINES = 5  # cap per day to reduce embedding dilution


def fetch_headlines_for_date(target_date: date, top_k: int = TOP_K_HEADLINES) -> list[str]:
    """
    Fetches up to top_k headlines relevant to broad U.S. equity market
    for a given trading date. Returns list of headline strings.

    NOTE: Returns empty list for dates with no coverage (pre-2018 gap).
    """
    if not ALPACA_API_KEY:
        raise EnvironmentError("ALPACA_API_KEY not set in environment.")

    params = {
        "symbols": ",".join(MARKET_SYMBOLS),
        "start": target_date.isoformat() + "T09:30:00Z",
        "end":   target_date.isoformat() + "T16:00:00Z",
        "limit": 20,
        "sort": "desc",
        "include_content": "false",
    }
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }

    resp = requests.get(ALPACA_NEWS_URL, params=params, headers=headers, timeout=10)
    if resp.status_code != 200:
        return []  # treat API failure as no news

    articles = resp.json().get("news", [])

    # Sort by relevance score if available; fall back to recency order
    articles.sort(key=lambda a: a.get("relevance_score", 0), reverse=True)

    headlines = [a["headline"] for a in articles[:top_k]]
    return headlines


def build_headline_cache(
    trading_dates: list[date],
    cache_path: str = "data/headline_cache.json",
    sleep_s: float = 0.5,
) -> dict:
    """
    Fetches and caches headlines for all trading dates.
    Skips dates already in the cache (resumable).

    Returns dict: { "YYYY-MM-DD": [headline, ...], ... }
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    else:
        cache = {}

    missing = [d for d in trading_dates if d.isoformat() not in cache]
    print(f"Fetching headlines for {len(missing)} dates (already cached: {len(cache)})...")

    for i, d in enumerate(missing):
        key = d.isoformat()
        try:
            headlines = fetch_headlines_for_date(d)
            cache[key] = headlines
        except Exception as e:
            print(f"  [WARN] {key}: {e}")
            cache[key] = []  # zero-impute on error

        if (i + 1) % 50 == 0:
            # Save checkpoint every 50 dates
            with open(cache_path, "w") as f:
                json.dump(cache, f)
            print(f"  Checkpoint saved at {i + 1}/{len(missing)}")

        time.sleep(sleep_s)  # rate limiting

    with open(cache_path, "w") as f:
        json.dump(cache, f)

    n_empty = sum(1 for v in cache.values() if not v)
    print(f"Done. {n_empty}/{len(cache)} dates have zero headlines (will be zero-imputed).")
    return cache


def load_headline_cache(cache_path: str = "data/headline_cache.json") -> dict:
    with open(cache_path) as f:
        return json.load(f)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="OHLCV CSV path")
    parser.add_argument("--cache", type=str, default="data/headline_cache.json")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    trading_dates = [d.date() for d in df.index]

    build_headline_cache(trading_dates, cache_path=args.cache)
