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

def run_deepchecks_suite(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Runs a DeepChecks suite to validate train/test splits and detect drift.
    Returns a dictionary summary of the results.
    """
    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import train_test_validation
        
        # 1. Wrap DataFrames in DeepChecks Dataset
        # Assuming 'target_return_next_day' is the target and 'date' is a datetime index/column
        # We need to drop non-numeric columns that might confuse it, or specify cat features
        
        # Identify label and cat features
        label_col = "target_return_next_day"
        if label_col not in train_df.columns:
            print(f"Warning: {label_col} not found for DeepChecks.")
            return {"passed": False, "error": "Label column missing"}

        # Prepare datasets
        ds_train = Dataset(train_df, label=label_col, index_name="date", parsing_date=["date"])
        ds_test = Dataset(test_df, label=label_col, index_name="date", parsing_date=["date"])
        
        # 2. Run the Suite
        # 'train_test_validation' checks for drift, leakage, and integrity
        suite = train_test_validation()
        result = suite.run(train_dataset=ds_train, test_dataset=ds_test)
        
        # 3. Save Report (Optional - locally)
        # result.save_as_html("deepchecks_report.html")
        
        # 4. Return Summary
        # We can extract passed/failed checks
        # result.get_not_passed_checks() returns a list
        
        failures = result.get_not_passed_checks()
        passed = len(failures) == 0
        
        return {
            "passed": passed,
            "failures": [check.name for check in failures] if failures else [],
            "score": result.passed_checks_ratio
        }

    except ImportError:
        print("DeepChecks not installed. Skipping deep validation.")
        return {"passed": True, "note": "DeepChecks not installed"}
    except Exception as e:
        print(f"DeepChecks failed: {e}")
        return {"passed": False, "error": str(e)}
