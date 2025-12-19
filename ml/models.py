import pickle
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from .config import MODELS_DIR, RF_N_ESTIMATORS, RF_MAX_DEPTH, CLUSTERS_K, PCA_COMPONENTS

def train_models(df: pd.DataFrame):
    """
    Trains Regression, Classification, PCA, and KMeans models.
    """
    # Features to use
    features = [
        "return_lag1", "return_lag2", "return_lag3", "return_lag5",
        "volatility_5d", "volatility_20d", "price_vs_ma20"
    ]
    
    X = df[features]
    y_reg = df["target_return_next_day"]
    y_clf = df["risk_class"]
    
    # Regression
    print("Training Regressor...")
    # [IMPROVEMENT] Use GradientBoostingRegressor.
    # Gradient Boosting often performs better than Random Forest on tabular data with subtle signals.
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Using GradientBoostingRegressor - often squeezes out better R2 than RF on noisy data
    regressor = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    regressor.fit(X, y_reg)
    
    # Classification
    print("Training Classifier (Gradient Boosting)...")
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Gradient Boosting is often superior for tabular data where decision boundaries are non-linear but smooth.
    # Tuned for ~60-65% accuracy without overfitting.
    classifier = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10 # Early stopping to prevent overfitting
    )
    classifier.fit(X, y_clf)
    
    # PCA & Clustering (Unsupervised)
    # We use the same features to cluster stock behaviors
    print("Training PCA & KMeans...")
    pca = PCA(n_components=PCA_COMPONENTS)
    X_pca = pca.fit_transform(X)
    
    kmeans = KMeans(n_clusters=CLUSTERS_K, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    
    return {
        "regressor": regressor,
        "classifier": classifier,
        "pca": pca,
        "kmeans": kmeans,
        "features": features # Save list of features to ensure consistency
    }

def save_models(models: dict) -> str:
    """
    Saves models to models/version_<timestamp>.
    Returns the version string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = MODELS_DIR / f"version_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        with open(save_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)
            
    print(f"Models saved to {save_dir}")
    return  f"version_{timestamp}"

def load_latest_models() -> dict:
    """
    Loads the most recent model version.
    """
    if not MODELS_DIR.exists():
        raise FileNotFoundError("Models directory not found.")
        
    versions = sorted([d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("version_")])
    
    if not versions:
        raise FileNotFoundError("No model versions found.")
        
    latest_version = versions[-1]
    print(f"Loading models from {latest_version}...")
    
    models = {}
    for filename in ["regressor.pkl", "classifier.pkl", "pca.pkl", "kmeans.pkl", "features.pkl"]:
        p = latest_version / filename
        if p.exists():
            with open(p, "rb") as f:
                models[p.stem] = pickle.load(f)
                
    return models

