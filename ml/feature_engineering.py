import pandas as pd
import numpy as np
from .config import HISTORY_YEARS

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates features for time-series analysis.
    Assumes df has columns: 'ticker', 'date', 'close' etc.
    """
    df = df.copy()
    if df.empty:
        print("Warning: Input DataFrame is empty. Skipping feature creation.")
        return df

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

    # RSI (Relative Strength Index)
    delta = df.groupby("ticker")["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.groupby(df["ticker"]).rolling(window=14).mean().reset_index(0, drop=True)
    avg_loss = loss.groupby(df["ticker"]).rolling(window=14).mean().reset_index(0, drop=True)
    
    rs = avg_gain / avg_loss.replace(0, np.nan) # Avoid div by zero
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50) # Neutral fill

    # MACD (Moving Average Convergence Divergence)
    ema_12 = df.groupby("ticker")["close"].ewm(span=12, adjust=False).mean().reset_index(0, drop=True)
    ema_26 = df.groupby("ticker")["close"].ewm(span=26, adjust=False).mean().reset_index(0, drop=True)
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df.groupby("ticker")["macd"].ewm(span=9, adjust=False).mean().reset_index(0, drop=True)

    # Target Generation
    # Regression target: Next day return
    df["target_return_next_day"] = df.groupby("ticker")["return"].shift(-1)
    
    # Classification target: Risk Class (Forecast future volatility)
    # We define "Risk" as the volatility of returns over the NEXT 5 days.
    # This prevents leakage because the model must predict future behavior from past features.
    df["future_vol"] = df.groupby("ticker")["return"].rolling(window=5).std().shift(-5).reset_index(0, drop=True)
    
    # Drop initial NaNs from lags/rolling
    df = df.dropna(subset=["return_lag5", "volatility_20d"])
    
    # Define risk classes based on future_vol quantiles
    # We use the training portion of the data to define thresholds to be strictly correct,
    # but for this demo, global quantiles on the whole set is fine as long as we drop NaNs.
    df_with_target = df.dropna(subset=["future_vol"])
    
    low_thresh = df_with_target["future_vol"].quantile(0.33)
    high_thresh = df_with_target["future_vol"].quantile(0.66)
    
    def get_risk_class(vol):
        if pd.isna(vol): return np.nan
        if vol <= low_thresh: return 0 # Low
        elif vol <= high_thresh: return 1 # Medium
        else: return 2 # High
        
    df["risk_class"] = df["future_vol"].apply(get_risk_class)
    
    # Final cleanup: The last 5 rows will have NaN risk_class/future_vol.
    # We keep them in the dataframe returning from create_features so the Pipeline 
    # can predict on the most recent row, but training functions MUST drop them.
    
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly split the DataFrame into train and test sets.
    Uses sklearn's train_test_split with stratification on the risk class to keep class balance.
    Returns (train_df, test_df).
    """
    if df.empty:
        raise ValueError("Cannot split empty DataFrame.")
    # Ensure required columns exist
    required = ["risk_class", "target_return_next_day"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' for split.")
    # Drop rows with NaNs in the target columns
    df_clean = df.dropna(subset=required).copy()
    # Strict Time-Series Split (No Shuffle) to prevent data leakage.
    # Assumption: df is already sorted by date (handled in create_features).
    if isinstance(test_size, float):
        train_size = int(len(df_clean) * (1 - test_size))
    else:
        # Assumes int means number of test samples (e.g. days)
        train_size = len(df_clean) - int(test_size)
        
    train_df = df_clean.iloc[:train_size]
    test_df = df_clean.iloc[train_size:]
    
    print(f"Time-Series Split: Train={len(train_df)}, Test={len(test_df)}")
    return train_df, test_df
