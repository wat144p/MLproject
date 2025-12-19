import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score
from .config import EXPERIMENTS_DIR
from datetime import datetime

def evaluate_models(models: dict, df_test: pd.DataFrame) -> dict:
    """
    Evaluates trained models on test data.
    """
    features = models["features"]
    X_test = df_test[features]
    y_reg_test = df_test["target_return_next_day"]
    y_clf_test = df_test["risk_class"]
    
    # Regression metrics
    y_reg_pred = models["regressor"].predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    r2 = r2_score(y_reg_test, y_reg_pred)
    
    # Classification metrics
    y_clf_pred = models["classifier"].predict(X_test)
    acc = accuracy_score(y_clf_test, y_clf_pred)
    f1 = f1_score(y_clf_test, y_clf_pred, average="weighted")
    precision = precision_score(y_clf_test, y_clf_pred, average="weighted", zero_division=0)
    recall = recall_score(y_clf_test, y_clf_pred, average="weighted", zero_division=0)
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "regression": {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        },
        "classification": {
            "Accuracy": acc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall
        }
    }
    
    # Save metrics
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(EXPERIMENTS_DIR / f"metrics_{timestamp}.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Metrics saved to experiments/metrics_{timestamp}.json")
    return metrics


