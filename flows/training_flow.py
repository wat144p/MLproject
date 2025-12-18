import sys
from pathlib import Path
# Add project root to python path to allow imports from 'ml'
sys.path.append(str(Path(__file__).parent.parent))

from prefect import flow, task, get_run_logger
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from ml.config import TICKERS, HISTORY_YEARS, TEST_SIZE_DAYS
from ml.data_ingestion import fetch_stock_data
from ml.feature_engineering import create_features, split_data
from ml.models import train_models, save_models
from ml.evaluation import evaluate_models
from ml.drift import check_data_integrity, check_feature_drift

@task(retries=3)
def get_data_task():
    return fetch_stock_data(TICKERS)

@task
def feature_engineering_task(df):
    return create_features(df)

@task
def validate_data_task(df):
    report = check_data_integrity(df)
    if not report["passed"]:
        print(f"Data integrity warning: {report}")
    return report

@task
def split_data_task(df):
    # Retrieve TEST_SIZE_DAYS from config. Note: config defines it as "days" (int), 
    # but split_data expects "test_size" (float ratio) or int count. 
    # If TEST_SIZE_DAYS is an integer (e.g. 30), sklearn interprets it as absolute number of samples.
    # We must ensure we pass it to the parameter 'test_size' that the function accepts.
    return split_data(df, test_size=TEST_SIZE_DAYS)

@task(name="notify_completion")
def notify_completion(version: str):
    """
    Sends a notification upon successful completion.
    In a real scenario, this would send a Slack message or Email.
    """
    message = f"✅ Pipeline Completed Successfully! Model Version: {version}"
    print(f"\n[NOTIFICATION] {message}\n")
    # Simulation of sending request
    # requests.post(SLACK_WEBHOOK_URL, json={"text": message})

@task
def train_task(train_df):
    return train_models(train_df)

@task
def evaluate_task(models, test_df):
    return evaluate_models(models, test_df)

@task
def save_task(models, metrics):
    # Save only the model artifacts; metrics are not persisted in this demo
    return save_models(models)

@flow(name="Stock Risk Training Flow")
def training_flow():
    """
    Orchestrates the ML pipeline:
    1. Ingestion
    2. Validation
    3. Feature Engineering
    4. Split
    5. Training
    6. Evaluation
    7. Saving
    8. Notification
    """
    logger = get_run_logger()
    logger.info("Starting training flow...")
    
    # 1. Get Data
    try:
        raw_df = get_data_task()
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return

    # 2. Validation
    validation_results = validate_data_task(raw_df)
    if not validation_results["passed"]:
        logger.warning(f"Data integrity warning: {validation_results}")
    
    # 3. Features
    df_features = feature_engineering_task(raw_df)
    
    if df_features.empty:
        logger.warning("No data available for training. Stopping flow.")
        return

    # 4. Split
    train_df, test_df = split_data_task(df_features)
    
    # 5. Drift Check (DeepChecks)
    try:
        from ml.drift import run_deepchecks_suite
        deepcheck_results = run_deepchecks_suite(train_df, test_df)
        if not deepcheck_results["passed"]:
             logger.warning(f"DeepChecks Drift/integrity warning: {deepcheck_results}")
        else:
             logger.info(f"DeepChecks Passed. Score: {deepcheck_results.get('score', 'N/A')}")
    except Exception as e:
        logger.error(f"DeepChecks failed to run: {e}")

    # 6. Train
    models = train_task(train_df)
    
    # 7. Evaluate
    metrics = evaluate_task(models, test_df)
    
    # 8. Save
    version = save_task(models, metrics)
    
    # 9. Notify
    notify_completion(version)
    
    logger.info(f"Flow completed. New model version: {version}")

if __name__ == "__main__":
    training_flow()

