import os
import json
import pandas as pd
from ml.models import load_latest_models
from ml.data_ingestion import fetch_stock_data
from ml.feature_engineering import create_features, split_data
from ml.evaluation import evaluate_models
from ml.config import TICKERS, HISTORY_YEARS, TEST_SIZE_DAYS, EXPERIMENTS_DIR

def get_latest_metrics(skip_recent=0):
    # Find the most recent metrics file in experiments directory
    if not EXPERIMENTS_DIR.exists():
        return None
    metric_files = sorted([f for f in EXPERIMENTS_DIR.iterdir() if f.is_file() and f.name.startswith('metrics_')])
    if len(metric_files) <= skip_recent:
        return None
    # We skip the most recent one if we just created a training metrics file
    latest = metric_files[-(1 + skip_recent)]
    with open(latest) as f:
        return json.load(f)

def main():
    # Load models
    models = load_latest_models()
    if not models:
        print('No models found.')
        return
    # Ingest data and create features
    df_raw = fetch_stock_data(TICKERS, use_cache=True)
    df_feat = create_features(df_raw)
    
    # Split to get training set
    # Note: split_data now takes test_size as a float (e.g. 0.2)
    train_df, _ = split_data(df_feat, test_size=0.2)
    
    # Evaluate on training data
    # IMPORTANT: evaluate_models saves metrics to disk. 
    train_metrics_output = evaluate_models(models, train_df)
    
    # Load actual test metrics (the ones saved from the pipeline run)
    # Since evaluate_models just saved a file, we skip the most recent one.
    test_metrics = get_latest_metrics(skip_recent=1)
    if not test_metrics:
        print('No test metrics file found.')
        return
    # Compare regression RMSE and classification accuracy
    train_rmse = train_metrics['regression']['RMSE']
    test_rmse = test_metrics['regression']['RMSE']
    train_acc = train_metrics['classification']['Accuracy']
    test_acc = test_metrics['classification']['Accuracy']
    print('--- Overfitting check ---')
    print(f'Training RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}')
    print(f'Training Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    # Simple heuristic: if training error is <70% of test error (or accuracy > test by >10%)
    if train_rmse < 0.7 * test_rmse:
        print('⚠️ Regression model may be overfitting (training error much lower than test error).')
    if train_acc > test_acc + 0.10:
        print('⚠️ Classification model may be overfitting (training accuracy significantly higher than test accuracy).')
    else:
        print('✅ No strong signs of overfitting detected.')

if __name__ == '__main__':
    main()
