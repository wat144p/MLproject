import pandas as pd
import numpy as np
from ml.data_ingestion import fetch_stock_data
from ml.feature_engineering import create_features
from ml.config import RISK_LEVELS, TICKERS

class PredictionService:
    def __init__(self, models):
        self.models = models
        self.features_list = models["features"]

    def _get_latest_features(self, ticker: str):
        # Fetch data (cached if possible/recent)
        # In production this would use a real feature store
        df = fetch_stock_data([ticker], use_cache=True)
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
            
        df_features = create_features(df)
        
        # Get the very last row
        latest = df_features.iloc[[-1]].copy()
        
        # Check if we have enough data (create_features might drop if not enough history)
        if latest.empty or latest[self.features_list].isna().any().any():
             # Fallback: maybe we need to fetch more data or handle new ticker
             # Try using the row before if the last one is NaN (e.g. next_day_return is NaN but lags are ok)
             # Actually create_features keeps NaN targets but lags should be present if history > 20 days
             if latest.iloc[0]["return_lag1"] is None or np.isnan(latest.iloc[0]["return_lag1"]):
                 raise ValueError("Not enough history to generate features.")
        
        return latest[self.features_list]

    def predict_risk(self, ticker: str):
        features = self._get_latest_features(ticker)
        
        # Proba
        probas = self.models["classifier"].predict_proba(features)[0]
        max_idx = np.argmax(probas)
        risk_class = RISK_LEVELS[max_idx] if max_idx < len(RISK_LEVELS) else "Unknown"
        
        return {
            "risk_class": risk_class,
            "probabilities": {RISK_LEVELS[i]: float(p) for i, p in enumerate(probas)}
        }

    def predict_return(self, ticker: str):
        features = self._get_latest_features(ticker)
        pred = self.models["regressor"].predict(features)[0]
        return float(pred)

    def recommend_similar(self, input_ticker: str, risk_preference: str = None):
        # 1. Get features for input
        input_features = self._get_latest_features(input_ticker)
        
        # 2. Get PCA projection
        pca_vec = self.models["pca"].transform(input_features)
        input_cluster = self.models["kmeans"].predict(pca_vec)[0]
        
        # 3. Find others in the same cluster
        # Only searching within our configured TICKERS for this demo
        recommendations = []
        for t in TICKERS:
            if t == input_ticker: 
                continue
                
            try:
                # Get features for candidate
                cand_features = self._get_latest_features(t)
                cand_pca = self.models["pca"].transform(cand_features)
                cand_cluster = self.models["kmeans"].predict(cand_pca)[0]
                
                if cand_cluster == input_cluster:
                    # Check risk if preference is set
                    # We need to predict risk for candidate
                    risk_info = self.predict_risk(t)
                    candidate_risk = risk_info["risk_class"]
                    
                    if risk_preference and risk_preference.lower() != candidate_risk.lower():
                        continue
                        
                    recommendations.append({
                        "ticker": t,
                        "risk_class": candidate_risk,
                        "cluster": int(cand_cluster) # cast to int for JSON
                    })
                    
            except Exception:
                continue
                
        return recommendations
