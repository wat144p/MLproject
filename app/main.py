from fastapi import FastAPI, HTTPException, Depends
from app.schemas import (
    RiskPredictionRequest, RiskPredictionResponse,
    ReturnPredictionRequest, ReturnPredictionResponse,
    RecommendationRequest, RecommendationResponse
)
from app.dependencies import get_models
from app.services import PredictionService
from ml.config import EXPERIMENTS_DIR, TICKERS
import json
import os

app = FastAPI(
    title="Stock Risk Forecasting API",
    description="End-to-End MLOps Project for Stock Risk Classification and Forecasting",
    version="1.0.0"
)

@app.get("/")
def home(models = Depends(get_models)):
    model_status = "Loaded" if models else "Not Loaded (Run training first)"
    return {
        "project": "Stock Risk Forecasting & Recommendation System",
        "domain": "Economics & Finance",
        "endpoints": [
            "/predict_risk",
            "/predict_return",
            "/recommend_similar",
            "/metrics"
        ],
        "model_status": model_status,
        "supported_tickers": TICKERS
    }

@app.post("/predict_risk", response_model=RiskPredictionResponse)
def predict_risk(request: RiskPredictionRequest, models = Depends(get_models)):
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    service = PredictionService(models)
    try:
        result = service.predict_risk(request.ticker)
        return {
            "ticker": request.ticker,
            "risk_class": result["risk_class"],
            "probabilities": result["probabilities"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_return", response_model=ReturnPredictionResponse)
def predict_return(request: ReturnPredictionRequest, models = Depends(get_models)):
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    service = PredictionService(models)
    try:
        pred = service.predict_return(request.ticker)
        return {
            "ticker": request.ticker,
            "predicted_next_day_return": pred
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend_similar") # GET for simpler query
def recommend_similar(ticker: str, risk_preference: str = None, models = Depends(get_models)):
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    service = PredictionService(models)
    try:
        recs = service.recommend_similar(ticker, risk_preference)
        return {
            "input_ticker": ticker,
            "recommendations": recs
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    # Find latest metrics file
    if not EXPERIMENTS_DIR.exists():
         raise HTTPException(status_code=404, detail="No experiments found")
         
    files = sorted(EXPERIMENTS_DIR.glob("metrics_*.json"))
    if not files:
        raise HTTPException(status_code=404, detail="No metrics found")
        
    latest = files[-1]
    with open(latest, "r") as f:
        data = json.load(f)
    return data
