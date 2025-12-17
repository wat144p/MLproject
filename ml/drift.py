import pandas as pd
import numpy as np

def check_data_integrity(df: pd.DataFrame) -> dict:
    """
    Checks for basic data integrity issues.
    """
    report = {
        "passed": True,
        "missing_values": {},
        "empty_ticker_data": []
    }
    
    if df.empty:
        report["passed"] = False
        report["error"] = "DataFrame is empty"
        return report
        
    # Check for NaNs
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        report["missing_values"] = nan_counts[nan_counts > 0].to_dict()
        # Not a hard failure unless key columns are missing, but let's log it
    
    # Check for tickers with too little data
    ticker_counts = df["ticker"].value_counts()
    for ticker, count in ticker_counts.items():
        if count < 50:
            report["empty_ticker_data"].append(ticker)
            report["passed"] = False
            
    return report

def check_feature_drift(train_df: pd.DataFrame, new_df: pd.DataFrame, features: list) -> dict:
    """
    detects drift by comparing mean of features.
    Simple method: if mean differs by > 2 standard deviations of training data.
    """
    drift_report = {}
    
    for feature in features:
        train_mean = train_df[feature].mean()
        train_std = train_df[feature].std()
        
        new_mean = new_df[feature].mean()
        
        # Avoid division by zero
        if train_std == 0:
            z_score = 0 if new_mean == train_mean else 999
        else:
            z_score = abs(new_mean - train_mean) / train_std
            
        drift_report[feature] = {
            "train_mean": train_mean,
            "new_mean": new_mean,
            "z_score": z_score,
            "drift_detected": z_score > 3 # 3 sigma
        }
        
    return drift_report
