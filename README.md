# End-to-End Stock Risk Forecasting & Recommendation System (MLOps)

**Student Name:** Muhammad Shayan Asif  
**Registration Number:** 2023909  
**Course Code:** AI-321  
**Domain:** Economics & Finance  

---

## 🚀 Overview
This project implements a production-grade MLOps pipeline for financial stock market analysis. It automates the entire lifecycle from data ingestion to containerized deployment.

### Key Features
- **Data Ingestion**: Multi-source fetching from Alpha Vantage with an automatic `yfinance` fallback strategy.
- **Advanced Feature Engineering**: Calculates technical indicators (Volatility, RSI, Lags, etc.).
- **Multi-Task ML Pipeline**:
    - **Classification**: Risk levels (Low, Medium, High).
    - **Regression**: Next-day price return forecasting.
    - **Clustering**: PCA + K-Means to identify similar market-behaving stocks.
- **Prefect Orchestration**: Fully automated and retriable training flows.
- **DeepChecks Validation**: Integrated data integrity and distribution drift detection.
- **Dockerized Deployment**: FastAPI service served via high-performance containers.
- **CI/CD**: Automated GitHub Actions for testing and image builds.

---

## 🛠️ Step-by-Step Deployment Guide

### 1. Environment Setup
Clone the repository and navigate to the project root.

**Create a Virtual Environment:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Install Dependencies:**
If `pip` is not recognized, use the python module execution:
```powershell
python -m pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory:
```env
ALPHAVANTAGE_API_KEY=your_key_here
```

### 3. Execution
**Run the Training Pipeline (Prefect):**
This will fetch data, run DeepChecks, train all models, and save versioned artifacts.
```powershell
# Recommended way (running as a module from root)
python -m flows.training_flow
```

**Launch the FastAPI Server:**
```powershell
uvicorn app.main:app --reload
```
View the dashboard at: [http://localhost:8000](http://localhost:8000)

### 4. Containerization
**Build the Docker Image:**
```bash
docker build -t stock-mlops-api .
```

**Run via Docker Compose:**
```bash
docker-compose up --build
```

---

## 🧪 Automated Testing
Run the full test suite (Integrity, Unit, and ML tests):
```bash
pytest
```

## 📈 System Architecture
1. **Data Layer**: Alpha Vantage API (with yfinance fallback).
2. **Orchestration Layer**: Prefect pipeline (Data -> Validation -> Training -> Versioning).
3. **Application Layer**: FastAPI (Async endpoints for inference).
4. **DevOps Layer**: Docker + GitHub Actions (CI/CD).
