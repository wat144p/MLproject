from prefect import flow, task
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
    return split_data(df, test_days=TEST_SIZE_DAYS)

@task
def train_task(train_df):
    return train_models(train_df)

@task
def evaluate_task(models, test_df):
    return evaluate_models(models, test_df)

@task
def save_task(models):
    return save_models(models)

@flow(name="Stock Risk Training Flow")
def training_flow():
    # 1. Ingestion
    raw_df = get_data_task()
    
    # 2. Validation
    validate_data_task(raw_df)
    
    # 3. Features
    df_features = feature_engineering_task(raw_df)
    
    # 4. Split
    train_df, test_df = split_data_task(df_features)
    
    # 5. Drift Check (vs recent) - just logging
    # In a real scenario we might compare against a reference dataset
    
    # 6. Train
    models = train_task(train_df)
    
    # 7. Evaluate
    evaluate_task(models, test_df)
    
    # 8. Save
    version = save_task(models)
    print(f"Flow completed. New model version: {version}")

if __name__ == "__main__":
    training_flow()
