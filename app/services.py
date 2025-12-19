import pandas as pd
import numpy as np
from ml.data_ingestion import fetch_stock_data
from ml.feature_engineering import create_features
from ml.config import RISK_LEVELS, TICKERS

class PredictionService:
    def __init__(self, models):
        self.models = models
        self.features_list = models["features"]

    def _get_latest_features(self, ticker: str, return_all=False):
        # Fetch data (cached if possible/recent)
        df = fetch_stock_data([ticker], use_cache=True)
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
            
        df_features = create_features(df)
        
        # Get the very last row
        latest = df_features.iloc[[-1]].copy()
        
        # Check if we have enough data
        if latest.empty:
             raise ValueError("Not enough history to generate features.")

        # If model features are NaN (e.g. not enough history for lags), warn/fail
        # But we try to proceed if return_all is True (for metadata extraction)
        if latest[self.features_list].isna().any().any():
             # Basic check: if critical lags are missing, we can't predict
             if latest.iloc[0]["return_lag1"] is None or np.isnan(latest.iloc[0]["return_lag1"]):
                 raise ValueError("Not enough history to generate features.")
        
        if return_all:
            return latest

        return latest[self.features_list]

    def predict_risk(self, ticker: str):
        # Get full row to extract volatility
        full_row = self._get_latest_features(ticker, return_all=True)
        features = full_row[self.features_list]
        
        # Proba
        probas = self.models["classifier"].predict_proba(features)[0]
        max_idx = np.argmax(probas)
        confidence = float(probas[max_idx])
        risk_class = RISK_LEVELS[max_idx] if max_idx < len(RISK_LEVELS) else "Unknown"
        
        # Extract Volatility (handle if missing)
        vol = 0.0
        if "volatility_20d" in full_row.columns:
            vol = float(full_row.iloc[0]["volatility_20d"])
            if np.isnan(vol): vol = 0.0

        # Simple Recommendation Logic
        # If Low Risk -> Buy/Hold
        # If High Risk -> Sell
        rec = "HOLD"
        if risk_class == "Low": rec = "BUY"
        elif risk_class == "High": rec = "SELL"
        
        return {
            "risk_class": risk_class,
            "probabilities": {RISK_LEVELS[i]: float(p) for i, p in enumerate(probas)},
            "volatility": vol,
            "confidence_score": confidence,
            "recommendation": rec
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


