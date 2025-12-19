import pytest
import pandas as pd
import numpy as np
from ml.models import train_models
from ml.drift import check_data_integrity

def test_data_integrity_check():
    df = pd.DataFrame({
        "ticker": ["A", "B"],
        "price": [10.0, None] # NaN
    })
    
    report = check_data_integrity(df)
    assert report["passed"] is False # Failed because count < 50
    assert "price" in report["missing_values"]

def test_minimal_training_performance():
    # Synthetic data for training
    np.random.seed(42)
    n_samples = 200
    
    df = pd.DataFrame({
        "ticker": ["MOCK"] * n_samples,
        "return_lag1": np.random.normal(0, 0.01, n_samples),
        "return_lag2": np.random.normal(0, 0.01, n_samples),
        "return_lag3": np.random.normal(0, 0.01, n_samples),
        "return_lag5": np.random.normal(0, 0.01, n_samples),
        "volatility_5d": np.abs(np.random.normal(0.01, 0.005, n_samples)),
        "volatility_20d": np.abs(np.random.normal(0.02, 0.005, n_samples)),
        "price_vs_ma20": np.random.normal(0, 0.05, n_samples),
    })
    
    # Simple linear relationship for regression target
    df["target_return_next_day"] = 0.5 * df["return_lag1"] + np.random.normal(0, 0.005, n_samples)
    
    # Simple rule for classification
    df["risk_class"] = (df["volatility_20d"] > 0.02).astype(int) # 0 or 1
    
    models = train_models(df)
    
    # Check if models are trained
    assert models["regressor"] is not None
    assert models["classifier"] is not None
    
    # Verify overfitting/learning on training set (sanity check)
    acc = models["classifier"].score(df[models["features"]], df["risk_class"])
    assert acc > 0.7 # Should learn the simple rule easily


