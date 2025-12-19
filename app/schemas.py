from pydantic import BaseModel
from typing import List, Dict, Optional

class RiskPredictionRequest(BaseModel):
    ticker: str

class RiskPredictionResponse(BaseModel):
    ticker: str
    risk_class: str
    probabilities: Dict[str, float]
    volatility: float
    confidence_score: float
    recommendation: str
    
class ReturnPredictionRequest(BaseModel):
    ticker: str

class ReturnPredictionResponse(BaseModel):
    ticker: str
    predicted_next_day_return: float
    
class RecommendationRequest(BaseModel):
    ticker: str
    risk_preference: Optional[str] = None # "Low", "Medium", "High"

class RecommendationResponse(BaseModel):
    input_ticker: str
    recommendations: List[Dict[str, str]] # List of {ticker: "AAPL", risk_class: "Low"}

