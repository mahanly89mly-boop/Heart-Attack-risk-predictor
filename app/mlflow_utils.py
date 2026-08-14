"""
MLflow tracking for the *serving* side of the project.

Model *training* tracking/registration lives in scripts/register_model.py
(run once, offline, after training). This module logs lightweight,
per-request metrics (probability, latency, prediction) so you can watch
serving behavior drift over time in the MLflow UI, separate from the
training experiments.

Logging is best-effort: if MLflow or the tracking server is unavailable,
the API must keep serving predictions, so all failures here are caught and
logged instead of raised.
"""
from __future__ import annotations

import logging
import threading

from app import config

logger = logging.getLogger("heart_disease_api.mlflow")

_mlflow = None
_lock = threading.Lock()
_initialized = False


def _get_mlflow():
    """Lazily import mlflow so the API can still start without it if
    ENABLE_MLFLOW_LOGGING=false, e.g. in constrained environments."""
    global _mlflow
    if _mlflow is None:
        import mlflow  # imported here so app can boot even if mlflow isn't installed
        _mlflow = mlflow
    return _mlflow


def init_mlflow() -> None:
    global _initialized
    if not config.ENABLE_MLFLOW_LOGGING:
        logger.info("MLflow logging disabled (ENABLE_MLFLOW_LOGGING=false)")
        return
    try:
        mlflow = _get_mlflow()
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        _initialized = True
        logger.info(
            "MLflow tracking initialized (uri=%s, experiment=%s)",
            config.MLFLOW_TRACKING_URI,
            config.MLFLOW_EXPERIMENT_NAME,
        )
    except Exception:
        logger.exception("Failed to initialize MLflow; continuing without serving-side tracking")
        _initialized = False


def log_prediction_run(
    request_id: str,
    risk_probability: float,
    prediction: int,
    latency_ms: float,
    model_version: str,
) -> None:
    if not config.ENABLE_MLFLOW_LOGGING or not _initialized:
        return
    try:
        mlflow = _get_mlflow()
        with _lock:
            with mlflow.start_run(run_name=f"predict-{request_id[:8]}"):
                mlflow.set_tag("request_id", request_id)
                mlflow.set_tag("model_version", model_version)
                mlflow.log_metric("risk_probability", risk_probability)
                mlflow.log_metric("prediction", prediction)
                mlflow.log_metric("latency_ms", latency_ms)
    except Exception:
        logger.exception("MLflow logging failed for request %s", request_id)
