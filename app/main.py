"""
Heart Disease Risk Prediction API
==================================
FastAPI service wrapping a trained Keras neural network that estimates the
probability a patient has heart disease, given clinical and lifestyle
features. Includes prediction logging (SQLite) and MLflow tracking so
serving behavior can be monitored over time.

Run locally:
    uvicorn app.main:app --reload

Interactive docs:
    http://localhost:8000/docs

Monitoring dashboard:
    http://localhost:8000/dashboard
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app import config, database, mlflow_utils
from app.model_service import model_service
from app.schemas import HealthResponse, PatientData, PredictionResponse, StatsResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("heart_disease_api")

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description=(
        "Predicts the probability of heart disease from clinical and "
        "lifestyle features using a trained Keras neural network."
    ),
    version=config.MODEL_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup() -> None:
    database.init_db()
    mlflow_utils.init_mlflow()
    logger.info("Startup complete. Model ready: %s", model_service.is_ready)


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "heart-disease-risk-api",
        "docs": "/docs",
        "dashboard": "/dashboard",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok" if model_service.is_ready else "degraded",
        model_loaded=model_service.model is not None,
        encoder_loaded=model_service.encoder is not None,
        model_version=config.MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(patient: PatientData, background_tasks: BackgroundTasks):
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        result = model_service.predict(patient)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    # Logging (SQLite + MLflow) happens after the response is prepared so it
    # never adds latency to the caller.
    background_tasks.add_task(
        database.log_prediction,
        result.request_id,
        patient.model_dump(mode="json"),
        result.risk_probability,
        result.prediction,
        result.risk_label.value,
        result.latency_ms,
        result.model_version,
    )
    background_tasks.add_task(
        mlflow_utils.log_prediction_run,
        result.request_id,
        result.risk_probability,
        result.prediction,
        result.latency_ms,
        result.model_version,
    )

    return result


@app.get("/stats", response_model=StatsResponse, tags=["monitoring"])
def stats():
    return database.get_stats()


@app.get("/recent-predictions", tags=["monitoring"])
def recent_predictions(limit: int = 20):
    return database.get_recent_predictions(limit=limit)


@app.get("/dashboard", response_class=HTMLResponse, tags=["monitoring"])
def dashboard():
    from pathlib import Path

    dashboard_path = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
    return dashboard_path.read_text(encoding="utf-8")
