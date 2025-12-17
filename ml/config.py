import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# Tickers to track
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Model hyperparameters (simple for "laptop-scale")
RF_N_ESTIMATORS = 50
RF_MAX_DEPTH = 10
CLUSTERS_K = 3
PCA_COMPONENTS = 3

# Data settings
HISTORY_YEARS = 2 # Fetch last 2 years to keep it light
TEST_SIZE_DAYS = 30 # Last 30 days for testing
VAL_SIZE_DAYS = 30  # Previous 30 days for validation

# Risk thresholds (Classification)
# Example: 0=Low, 1=Medium, 2=High
RISK_LEVELS = ["Low", "Medium", "High"]
