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
    
    train, test = split_data(df, test_days=10)
    
    # Test set should loosely capture the last 10 days 
    # (strictly it captures all rows >= split_date)
    assert len(test) >= 10
    assert len(train) < 100
    # Provide gap for overlap or exact logic check (split is by date, 1 row per day)
    assert 80 <= len(train) <= 90
