
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import time
import requests  # Alpha Vantage API calls
import yfinance as yf  # fallback if Alpha Vantage fails
from pathlib import Path
from typing import List, Optional
from .config import DATA_DIR

# Get Alpha Vantage API key from environment variables
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

def fetch_stock_data(tickers: List[str], use_cache: bool = True) -> pd.DataFrame:
    """
    Fetches daily stock data for the given tickers using Alpha Vantage API.
    Returns a combined DataFrame with columns: [ticker, date, open, high, low, close, volume]
    """
    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set.")

    all_data = []
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        cache_path = DATA_DIR / f"{ticker}.csv"
        
        # Simple cache logic
        if use_cache and cache_path.exists():
            # Check if cache is stale (older than 24 hours)
            last_modified = cache_path.stat().st_mtime
            if (time.time() - last_modified) > 86400: # 24 hours in seconds
                print(f"Cache for {ticker} is expired (>24h). Refetching...")
            else:
                print(f"Loading {ticker} from cache...")
                try:
                    df = pd.read_csv(cache_path)
                    df["date"] = pd.to_datetime(df["date"], utc=True)
                    all_data.append(df)
                    continue
                except Exception:
                    print(f"Cache corrupted for {ticker}, refetching...")

        print(f"Fetching {ticker} from Alpha Vantage...")
        
        try:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
            response = requests.get(url)
            response.raise_for_status() # Raise an exception for HTTP errors
            data = response.json()

            # Check for premium or info messages from Alpha Vantage
            if "Information" in data and "premium" in data["Information"].lower():
                print(f"Alpha Vantage premium limit reached for {ticker}, falling back to yfinance")
                try:
                    yf_ticker = yf.Ticker(ticker)
                    hist = yf_ticker.history(period="5y")
                    if hist.empty:
                        print(f"yfinance returned empty data for {ticker}")
                        continue
                    hist = hist.reset_index().rename(columns={
                        "Date": "date",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    })
                    hist["date"] = pd.to_datetime(hist["date"], utc=True)
                    df = hist[["date", "open", "high", "low", "close", "volume"]].copy()
                except Exception as e:
                    print(f"yfinance error for {ticker}: {e}")
                    continue
                # Append and skip the rest of Alpha Vantage processing
                df["ticker"] = ticker
                df = df.sort_values("date")
                if use_cache:
                    df.to_csv(cache_path, index=False)
                all_data.append(df)
                # Respect rate limit (still wait a bit)
                time.sleep(12)
                continue
            # Debug: show the raw response when we don't get a time series
            if "Error Message" in data:
                print(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
                continue
            if "Note" in data:
                print(f"Alpha Vantage note for {ticker}: {data['Note']}")
                time.sleep(15)
                continue
            # If we reach here but there is no time series, dump the whole payload for inspection
            if "Time Series (Daily)" not in data:
                print(f"Unexpected Alpha Vantage response for {ticker}: {data}")
                continue
            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                print(f"Warning: No daily time series data found for {ticker}")
                continue

            # Convert the time series data to a DataFrame
            hist = pd.DataFrame.from_dict(time_series, orient="index")
            hist.index.name = "date"
            hist = hist.reset_index()

            # Rename columns to a standardized format
            # Alpha Vantage gives: 1. open, 2. high, 3. low, 4. close, 5. adjusted close, 6. volume, 7. dividend amount, 8. split coefficient
            hist.columns = [
                "date", "open", "high", "low", "close", "adjusted close",
                "volume", "dividend amount", "split coefficient"
            ]
            
            # Convert date column to datetime objects
            hist["date"] = pd.to_datetime(hist["date"], utc=True)

            # Standardize columns
            needed_cols = ["date", "open", "high", "low", "close", "volume"]
            # Ensure all needed columns exist and are numeric
            for col in needed_cols[1:]: # Skip 'date'
                hist[col] = pd.to_numeric(hist[col], errors='coerce')
            
            if not all(col in hist.columns for col in needed_cols):
                 print(f"Missing columns for {ticker}. Found: {hist.columns}")
                 continue

            df = hist[needed_cols].copy()
            df["ticker"] = ticker
            
            # Sort
            df = df.sort_values("date")
            
            # Cache it
            if use_cache:
                df.to_csv(cache_path, index=False)
            
            all_data.append(df)
            
            # Be nice to the API
            time.sleep(12)  # Respect free tier rate limit (≈5 calls/min)
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

