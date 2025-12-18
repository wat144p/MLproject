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
    # [IMPROVEMENT] Use GradientBoostingRegressor with Scaling.
    # Gradient Boosting often performs better than Random Forest on tabular data with subtle signals.
    # StandardScaler ensures features are on the same scale, assisting convergence.
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    # Using RandomForestRegressor with strict regularization.
    # Given the small dataset (compact mode = 100 days), preventing overfitting is key.
    regressor = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=200,       # More trees for stability
            max_depth=5,            # Shallow to avoid fitting noise
            min_samples_leaf=10,    # Require significant samples to make a decision
            max_features='sqrt',    # Decorrelate trees
            random_state=42,
            n_jobs=-1
        ))
    ])
    regressor.fit(X, y_reg)
    
    # Classification
    print("Training Classifier...")
    # Keeping RandomForest for classification as it is performing well (>0.98 accuracy reported)
    classifier = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, 
        max_depth=RF_MAX_DEPTH, 
        random_state=42,
        n_jobs=-1
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
