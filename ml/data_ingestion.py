import os
import requests
import pandas as pd
import time
from pathlib import Path
from typing import List, Optional
from .config import DATA_DIR

def fetch_stock_data(tickers: List[str], use_cache: bool = True) -> pd.DataFrame:
    """
    Fetches daily stock data for the given tickers from Alpha Vantage.
    Returns a combined DataFrame with columns: [ticker, date, open, high, low, close, volume]
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Environment variable ALPHAVANTAGE_API_KEY is not set.")

    all_data = []
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        cache_path = DATA_DIR / f"{ticker}.csv"
        
        if use_cache and cache_path.exists():
            print(f"Loading {ticker} from cache...")
            df = pd.read_csv(cache_path)
            df["date"] = pd.to_datetime(df["date"])
            all_data.append(df)
            continue

        print(f"Fetching {ticker} from Alpha Vantage...")
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": api_key,
            "outputsize": "full" # or 'compact' for last 100 days
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Rate limit handling (free tier is 5 requests/minute)
            # If we hit a limit, wait a bit. 
            if "Note" in data:
                 # Alpha Vantage API call frequency note
                 print(f"Rate limit note: {data['Note']}")
                 time.sleep(15) 

            ts_data = data.get("Time Series (Daily)")
            if not ts_data:
                print(f"Warning: No data found for {ticker}")
                continue

            # Convert JSON to DataFrame
            records = []
            for date_str, values in ts_data.items():
                records.append({
                    "ticker": ticker,
                    "date": date_str,
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "volume": float(values["5. volume"])
                })
            
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            
            # Cache it
            if use_cache:
                df.to_csv(cache_path, index=False)
            
            all_data.append(df)
            
            # Respect rate limits (simple sleep)
            time.sleep(12) 
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df
