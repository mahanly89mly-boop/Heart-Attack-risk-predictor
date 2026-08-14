# Heart Disease Risk Prediction

A production-style machine learning service that estimates a patient's
probability of heart disease from 25 clinical and lifestyle features,
served through a REST API with request logging, experiment tracking, and
a live monitoring dashboard.

<p>
  <img alt="python" src="https://img.shields.io/badge/python-3.11-blue">
  <img alt="fastapi" src="https://img.shields.io/badge/FastAPI-0.115-009688">
  <img alt="tensorflow" src="https://img.shields.io/badge/TensorFlow-2.17-FF6F00">
  <img alt="mlflow" src="https://img.shields.io/badge/MLflow-2.19-0194E2">
  <img alt="docker" src="https://img.shields.io/badge/Docker-ready-2496ED">
</p>

## Overview

The model is a Keras `Sequential` neural network (`31 → 16 → 8 → 1`,
ReLU/sigmoid, trained with `Adam` + early stopping) trained on a 9,000-row
clinical dataset with 6-fold stratified cross-validation. It was compared
against `RandomForestClassifier` and `GradientBoostingClassifier` baselines
and selected for deployment.

| Metric (held-out test split) | Value |
|---|---|
| Accuracy | 89.1% |
| Precision (heart disease) | 86.3% |
| Recall (heart disease) | 76.3% |
| F1-score (heart disease) | 0.81 |

This repository turns that trained model into a real service:

- **FastAPI** REST API (`/predict`, `/health`, `/stats`, `/dashboard`)
- **Docker** image for consistent, portable deployment
- **Render / Railway** ready-to-deploy configuration
- **MLflow** tracking — both training-run registration and live serving
  metrics (probability, latency, prediction) per request
- **Monitoring dashboard** (`/dashboard`) — prediction volume, risk
  distribution, latency, and recent predictions, auto-refreshing
- **SQLite** request log powering the dashboard, with zero extra
  infrastructure required

## Architecture

```
                     ┌───────────────────┐
   client  ───POST──▶│   FastAPI  app     │
                     │  /predict          │
                     └────────┬──────────┘
                              │
              ┌───────────────┼────────────────┐
              ▼                                 ▼
     ┌─────────────────┐               ┌──────────────────┐
     │ Keras model +    │               │  Background tasks │
     │ OneHotEncoder    │               │  (non-blocking)   │
     │ (artifacts/)     │               └─────────┬─────────┘
     └─────────────────┘                           │
                                    ┌────────────────┴────────────────┐
                                    ▼                                  ▼
                          ┌──────────────────┐              ┌──────────────────┐
                          │ SQLite log        │              │ MLflow tracking   │
                          │ (data/monitoring.db) │           │ (per-request run) │
                          └─────────┬──────────┘              └──────────────────┘
                                    │
                                    ▼
                         GET /dashboard (Chart.js UI)
```

## Project structure

```
heart-disease-risk-api/
├── app/
│   ├── main.py            # FastAPI app & routes
│   ├── model_service.py   # Model loading + preprocessing + inference
│   ├── schemas.py         # Pydantic request/response models
│   ├── database.py        # SQLite prediction logging for the dashboard
│   ├── mlflow_utils.py    # Serving-side MLflow tracking
│   └── config.py          # Environment-driven configuration
├── dashboard/
│   └── index.html         # Self-contained monitoring UI (Chart.js)
├── artifacts/
│   ├── model.pkl           # Trained Keras model
│   └── OneHotEncoder.pkl   # Fitted encoder for categorical features
├── scripts/
│   └── register_model.py  # Registers model+encoder as one MLflow model
├── Dockerfile
├── docker-compose.yml      # API + local MLflow server
├── requirements.txt
├── render.yaml              # Render deploy blueprint
├── Procfile                  # Railway/Heroku-style process file
├── .env.example
└── DEPLOYMENT.md             # Step-by-step deploy guide
```

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
uvicorn app.main:app --reload
```

- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8000/dashboard
- Health check: http://localhost:8000/health

## API usage

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 54, "sex": "Male",
    "resting_bp_systolic": 132, "resting_bp_diastolic": 84,
    "cholesterol_total": 221, "hdl": 45, "ldl": 130, "triglycerides": 150,
    "fasting_blood_sugar": 110, "hba1c": 5.8, "bmi": 27.4,
    "resting_heart_rate": 72, "max_heart_rate_achieved": 158,
    "chest_pain_type": "Atypical Angina", "exercise_induced_angina": false,
    "st_depression": 1.2, "family_history": true, "smoker_status": "Former",
    "alcohol_units_per_week": 3.5, "exercise_minutes_per_week": 90,
    "sleep_hours": 6.5, "stress_score": 6.2, "wearable_owner": true,
    "daily_steps": 6500, "diet_quality_score": 58.0
  }'
```

Response:

```json
{
  "prediction": 0,
  "risk_probability": 0.183,
  "risk_label": "low",
  "model_version": "1.0.0",
  "latency_ms": 4.221,
  "request_id": "b9b2e6b0-...",
  "timestamp": "2026-08-14T18:30:00.123456+00:00"
}
```

Full field descriptions, types, and validation ranges are in the
interactive docs at `/docs`.

## Docker

```bash
docker build -t heart-disease-risk-api .
docker run -p 8000:8000 --env-file .env heart-disease-risk-api
```

Or, to run the API together with a local MLflow tracking server:

```bash
docker compose up --build
```

- API: http://localhost:8000
- MLflow UI: http://localhost:5000

## MLflow tracking

Two separate, intentional uses of MLflow:

1. **Training-side (versioning):** `scripts/register_model.py` wraps the
   Keras model + `OneHotEncoder` into a single reproducible
   `mlflow.pyfunc` model, logs the recorded evaluation metrics, and
   registers it in the MLflow Model Registry as
   `heart-disease-risk-classifier`.

   ```bash
   python scripts/register_model.py
   ```

2. **Serving-side (live monitoring):** every `/predict` call logs its
   probability, prediction, and latency as an MLflow run in the
   `heart-disease-risk-serving` experiment (toggle with
   `ENABLE_MLFLOW_LOGGING`). This is separate from the SQLite-backed
   `/dashboard`, which is optimized for a fast, dependency-free UI;
   MLflow is there for deeper experiment-style analysis and audit trail.

   > Logging an MLflow run per request is fine for a portfolio/demo
   > deployment. For high-traffic production use, batch or sample the
   > logging instead (e.g. log every Nth request, or aggregate and log
   > every minute).

## Monitoring dashboard

`GET /dashboard` renders a live view (no build step, single HTML file)
showing:

- Total predictions, positive rate, average risk probability, average
  latency, predictions in the last 24h
- Predictions-per-hour trend
- Risk distribution (low / medium / high)
- A table of the most recent predictions

It reads from two JSON endpoints you can also consume programmatically:
`GET /stats` and `GET /recent-predictions`.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for a full step-by-step guide to
deploying this service on **Render** or **Railway**.

## Tech stack

Python · TensorFlow/Keras · scikit-learn · FastAPI · Pydantic · SQLite ·
MLflow · Docker · Chart.js
