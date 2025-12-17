import pandas as pd
import numpy as np
from .config import HISTORY_YEARS

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates features for time-series analysis.
    Assumes df has columns: 'ticker', 'date', 'close' etc.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"])

    # Calculate daily returns
    df["return"] = df.groupby("ticker")["close"].pct_change()
    
    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f"return_lag{lag}"] = df.groupby("ticker")["return"].shift(lag)
    
    # Rolling statistics
    # Volatility (Std Dev of returns)
    df["volatility_5d"] = df.groupby("ticker")["return"].rolling(window=5).std().reset_index(0, drop=True)
    df["volatility_20d"] = df.groupby("ticker")["return"].rolling(window=20).std().reset_index(0, drop=True)
    
    # Rolling mean
    df["ma_5d"] = df.groupby("ticker")["close"].rolling(window=5).mean().reset_index(0, drop=True)
    df["ma_20d"] = df.groupby("ticker")["close"].rolling(window=20).mean().reset_index(0, drop=True)
    
    # Relative to MA
    df["price_vs_ma20"] = (df["close"] - df["ma_20d"]) / df["ma_20d"]

    # Target Generation
    # Regression target: Next day return
    df["target_return_next_day"] = df.groupby("ticker")["return"].shift(-1)
    
    # Classification target: Risk Class
    # Based on recent volatility (e.g., last 20 days)
    # 0 = Low Risk, 1 = Medium, 2 = High
    # We'll use quantiles across the whole dataset for simplicity/robustness
    # or per ticker. Global quantiles make stocks comparable.
    
    # Drop initial NaNs from lags/rolling
    df = df.dropna(subset=["return_lag5", "volatility_20d"])
    
    # Define risk classes based on volatility_20d quantiles
    low_thresh = df["volatility_20d"].quantile(0.33)
    high_thresh = df["volatility_20d"].quantile(0.66)
    
    def get_risk_class(vol):
        if vol <= low_thresh: return 0 # Low
        elif vol <= high_thresh: return 1 # Medium
        else: return 2 # High
        
    df["risk_class"] = df["volatility_20d"].apply(get_risk_class)
    
    # Final cleanup (last row will have NaN target_return_next_day, keep it for inference? 
    # For training we drop it. For simple logic, we'll return the full DF 
    # but the training function should handle NaNs or we drop here.)
    # We'll drop rows where targets are NaN for training safety here, 
    # but strictly speaking for "latest prediction" we need the last row.
    # Let's keep the last row but users of this DF must handle NaNs if training.
    
    return df

def split_data(df: pd.DataFrame, test_days: int = 30):
    """
    Time-based split.
    Last `test_days` for test, rest for train.
    """
    dates = df["date"].unique()
    split_date = pd.to_datetime(dates).sort_values()[-test_days]
    
    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()
    
    # Drop NaNs in targets for training data
    train_df = train_df.dropna(subset=["target_return_next_day"])
    test_df = test_df.dropna(subset=["target_return_next_day"])

    return train_df, test_df
