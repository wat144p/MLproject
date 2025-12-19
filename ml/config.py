import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# Tickers to track
# Mix of US Tech and Pakistan Stock Exchange (PSX via .PA suffix)
TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "NVDA",
] # PSX Blue Chips

# Model hyperparameters (simple for "laptop-scale")
# Model hyperparameters (simple for "laptop-scale")
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 15
CLUSTERS_K = 3
PCA_COMPONENTS = 3

# Data settings
HISTORY_YEARS = 5 # Fetch last 5 years for more data
TEST_SIZE_DAYS = 90 # Last 90 days (approx 3 months) for testing
VAL_SIZE_DAYS = 30  # Previous 30 days for validation

# Risk thresholds (Classification)
# Example: 0=Low, 1=Medium, 2=High
RISK_LEVELS = ["Low", "Medium", "High"]

