from fastapi.testclient import TestClient
from app.main import app
from app.dependencies import get_models
from unittest.mock import MagicMock
import pandas as pd
import pytest

client = TestClient(app)

# Mock models
mock_models = {
    "regressor": MagicMock(),
    "classifier": MagicMock(),
    "pca": MagicMock(),
    "kmeans": MagicMock(),
    "features": ["f1"] # Dummy feature list
}

# Mock methods
mock_models["regressor"].predict.return_value = [0.01]
mock_models["classifier"].predict_proba.return_value = [[0.8, 0.1, 0.1]]
mock_models["pca"].transform.return_value = [[1.0, 0.0, 0.0]]
mock_models["kmeans"].predict.return_value = [0]

# Mock fetch_stock_data to avoid API calls
@pytest.fixture
def mock_fetch(mocker):
    # Return a DataFrame compatible with "create_features"
    # Actually "PredictionService" calls "create_features" so we need to mock "fetch_stock_data" return
    # and maybe "create_features" return to simplify testing "services" logic or mock the service entirely.
    # For integration test of API, mocking get_models is easiest if we can trust the Service logic,
    # but the Service logic calls fetch_stock_data.
    
    # Let's mock the "fetch_stock_data" inside "app.services" or "ml.data_ingestion"
    mock_df = pd.DataFrame([{ "ticker": "AAPL", "date": "2023-01-01", "close": 100.0, "volume": 1000 }])
    mocker.patch("app.services.fetch_stock_data", return_value=mock_df)
    
    # Also mock create_features to return a DataFrame with expected columns
    m_feat = pd.DataFrame([{ "f1": 1.0, "return_lag1": 0.01, "target_return_next_day": 0.01 }]) # etc
    mocker.patch("app.services.create_features", return_value=m_feat)
    
    # And mock the get_models dependency
    app.dependency_overrides[get_models] = lambda: mock_models
    yield
    app.dependency_overrides = {}

def test_home():
    response = client.get("/", follow_redirects=False)
    # It redirects to /static/index.html
    assert response.status_code in [307, 200] 
    if response.status_code == 200:
        # If it followed redirect
        assert "text/html" in response.headers["content-type"]

def test_predict_risk(mock_fetch):
    response = client.post("/predict_risk", json={"ticker": "AAPL"})
    # Since we mocked models, we expect success
    assert response.status_code == 200
    assert response.json()["ticker"] == "AAPL"
    assert "risk_class" in response.json()

def test_predict_return(mock_fetch):
    response = client.post("/predict_return", json={"ticker": "AAPL"})
    assert response.status_code == 200
    assert "predicted_next_day_return" in response.json()


