import pandas as pd
import pytest
from ml.feature_engineering import create_features, split_data

def test_feature_creation():
    # Create dummy data
    data = {
        "ticker": ["ABC"] * 50,
        "date": pd.date_range(start="2023-01-01", periods=50),
        "close": [100 + i for i in range(50)]
    }
    df = pd.DataFrame(data)
    
    # Run feature creation
    df_features = create_features(df)
    
    # Assertions
    assert "return_lag1" in df_features.columns
    assert "volatility_20d" in df_features.columns
    assert "risk_class" in df_features.columns
    assert not df_features.empty

def test_split_data():
    data = {
        "ticker": ["ABC"] * 100,
        "date": pd.date_range(start="2023-01-01", periods=100),
        "close": range(100),
        "target_return_next_day": range(100) # Mock target
    }
    df = pd.DataFrame(data)
    
    # test_split_data expects df to have 'risk_class' for stratification
    # and 'target_return_next_day' (which we added).
    # Also we must provide 'risk_class' because split_data uses stratify=df['risk_class']
    df["risk_class"] = 1 # Dummy class
    
    # split_data expects 'test_size' (float or int), not 'test_days'
    train, test = split_data(df, test_size=0.1)
    
    # Check split ratio approx
    assert len(test) == 10
    assert len(train) == 90

