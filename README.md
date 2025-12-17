# End-to-End Stock Risk Forecasting & Recommendation System

## Project Title
**End‑to‑End Stock Risk Forecasting & Recommendation System with FastAPI, Prefect, CI/CD, Automated Testing, and Docker**

## Overview
This project implements a complete MLOps pipeline for stock market analysis. It features:
- **Data Ingestion**: Fetches daily stock data from Alpha Vantage.
- **Feature Engineering**: Computes rolling statistics, volatility, and lag features.
- **Machine Learning**: 
    - **Regression**: Predicts next-day returns (RandomForestRegressor).
    - **Classification**: Categorizes stocks into Risk Levels (Low, Medium, High).
    - **Unsupervised Learning**: PCA and K-Means clustering for stock similarity.
- **Orchestration**: Prefect flow for reproducible training pipelines.
- **Deployment**: FastAPI application for real-time predictions and recommendations.
- **Monitorng & Quality**: Automated tests, data drift checks, and Docker containerization.

## Setup Instructions

### 1. Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional, for containerized run)
- Alpha Vantage API Key (Get a free key from [alphavantage.co](https://www.alphavantage.co/))

### 2. Installation
1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure Environment:
   - Copy `.env.example` to `.env` (or set env var directly).
   - Add your API Key: `ALPHAVANTAGE_API_KEY=your_key_here`.

### 3. Running the Training Pipeline
Run the Prefect flow to fetch data, train models, and save artifacts:
```bash
python scripts/run_training_locally.py
```
This will create a new model version in `models/version_<timestamp>`.

### 4. Running the API
Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```
Access the comprehensive API documentation at: http://127.0.0.1:8000/docs

### 5. Running Tests
Execute unit and integration tests:
```bash
pytest tests/
```

### 6. Docker Deployment
Build and run with Docker Compose:
```bash
docker-compose up --build
```
The API will be available at http://localhost:8000.

## Endpoints
- **POST** `/predict_risk`: Predict risk class (Low/Medium/High) for a ticker.
- **POST** `/predict_return`: Forecast next-day return.
- **GET** `/recommend_similar?ticker=AAPL`: Find stocks with similar market behavior.
- **GET** `/metrics`: View latest model performance metrics.

## CI/CD
A GitHub Actions workflow is included in `.github/workflows/ci_cd.yml` to automatically run tests and build the Docker image on push.
