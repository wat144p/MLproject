# System Architecture

## Overview
The system follows a modular microservices-ready architecture, separating the ML pipeline (Training) from the serving layer (Inference).

## Components

### 1. Data Layer
- **Source**: Alpha Vantage API (Time Series Daily).
- **Storage**: Local CSV cache in `data/` for efficiency and rate-limit handling.
- **Drift**: Simple statistical checks between training and new data.

### 2. ML Pipeline (Prefect)
Located in `flows/training_flow.py`, the orchestration pipeline executes:
1. **Ingestion**: `fetch_stock_data`
2. **Validation**: `check_data_integrity`
3. **Feature Engineering**: `create_features` (Lags, Rolling Volatility, MA)
4. **Splitting**: Time-based split (Train vs Test).
5. **Training**: 
   - RandomForestRegressor (Return Forecasting)
   - RandomForestClassifier (Risk Classification)
   - PCA + KMeans (Clustering/Recommendation)
6. **Evaluation**: Metrics calculation and logging to `experiments/`.
7. **Registration**: Saving versioned models to `models/`.

### 3. Inference Layer (FastAPI)
- **Model Loading**: Models are loaded into memory on startup (singleton pattern via Dependencies).
- **Logic**: `PredictionService` handles feature reconstruction for single-ticker inference.
    - It fetches the latest data for the requested ticker.
    - Re-computes features (rolling windows require recent history).
    - Feeds features into the loaded models.
- **Endpoints**: RESTful JSON endpoints.
    - `/predict_risk`: Classification probability.
    - `/recommend_similar`: Uses PCA embeddings to find nearest neighbors (Cluster-based).

### 4. Infrastructure
- **Docker**: Single container encapsulating the API and dependencies.
- **CI/CD**: GitHub Actions triggers pytest suite and Docker build verification.

