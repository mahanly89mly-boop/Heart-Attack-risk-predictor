"""
Central configuration, driven entirely by environment variables so the same
image can run locally, in Docker, or on Render/Railway without code changes.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Model artifacts -------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "artifacts" / "model.pkl"))
ENCODER_PATH = os.getenv("ENCODER_PATH", str(BASE_DIR / "artifacts" / "OneHotEncoder.pkl"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# --- Monitoring / logging ---------------------------------------------------
LOG_DB_PATH = os.getenv("LOG_DB_PATH", str(BASE_DIR / "data" / "monitoring.db"))

# --- MLflow -------------------------------------------------------------
# Any standard MLflow tracking URI: local folder ("file:./mlruns"), a remote
# MLflow server ("http://mlflow:5000"), or a managed MLflow (Databricks URI).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{BASE_DIR / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "heart-disease-risk-serving")
ENABLE_MLFLOW_LOGGING = os.getenv("ENABLE_MLFLOW_LOGGING", "true").lower() == "true"

# --- Risk thresholds ------------------------------------------------------
# Probability cut points used only to turn a raw probability into a
# human-readable label for the dashboard/API response. Adjust freely;
# they do not affect the underlying 0/1 prediction (which uses 0.5).
RISK_LOW_MAX = float(os.getenv("RISK_LOW_MAX", "0.3"))
RISK_MEDIUM_MAX = float(os.getenv("RISK_MEDIUM_MAX", "0.6"))

# --- CORS -------------------------------------------------------------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
