from ml.config import TICKERS
from ml.data_ingestion import fetch_stock_data
import os

print(f"Current CWD: {os.getcwd()}")
print(f"Fetching tickers: {TICKERS}")

try:
    df = fetch_stock_data(TICKERS, use_cache=False)
    if df.empty:
        print("!!! DF IS EMPTY !!!")
    else:
        print(f"Success! Shape: {df.shape}")
        print("Head:")
        print(df.head())
        print("Data dir content:")
        # Check data dir
        if os.path.exists("data"):
            print(os.listdir("data"))
        else:
            print("DATA DIR MISSING")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")

