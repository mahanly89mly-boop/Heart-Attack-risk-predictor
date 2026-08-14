"""
Registers the already-trained model (artifacts/model.pkl +
artifacts/OneHotEncoder.pkl) into MLflow as a single, reproducible
pyfunc model, together with the evaluation metrics recorded in the
training notebook.

This is the "training-side" MLflow usage: run it once (or whenever you
retrain) to version the model in the MLflow Model Registry. It is
separate from app/mlflow_utils.py, which logs live serving traffic.

Usage:
    python scripts/register_model.py

Environment variables (same as the API, see app/config.py):
    MLFLOW_TRACKING_URI   (default: file:./mlruns)
    MODEL_PATH             (default: artifacts/model.pkl)
    ENCODER_PATH            (default: artifacts/OneHotEncoder.pkl)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import config  # noqa: E402
from app.model_service import CATEGORICAL_FEATURE_ORDER, NUMERIC_FEATURE_ORDER  # noqa: E402

# Metrics recorded from the training notebook's held-out test split
# (see the classification_report / confusion_matrix cells). Update these
# if you retrain the model.
TRAINING_METRICS = {
    "test_accuracy": 0.8913,
    "train_accuracy": 0.8977,
    "precision_class1": 0.8632,
    "recall_class1": 0.7626,
    "f1_class1": 0.81,
    "true_negatives": 990,
    "false_positives": 55,
    "false_negatives": 108,
    "true_positives": 347,
}


class HeartDiseaseRiskModel(mlflow.pyfunc.PythonModel):
    """Wraps the Keras model + OneHotEncoder into one deployable unit so
    preprocessing always travels with the model."""

    def load_context(self, context):
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)
        with open(context.artifacts["encoder"], "rb") as f:
            self.encoder = pickle.load(f)

    def predict(self, context, model_input):
        df = model_input.copy()
        numeric = df[NUMERIC_FEATURE_ORDER].astype(np.float32).values
        categorical = self.encoder.transform(df[CATEGORICAL_FEATURE_ORDER]).astype(np.float32)
        features = np.hstack([numeric, categorical])
        return self.model.predict(features, verbose=0).reshape(-1)


def main() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("heart-disease-risk-training")

    with mlflow.start_run(run_name="keras-31-16-8-1-registration"):
        mlflow.log_params(
            {
                "architecture": "Dense(31)-Dense(16)-Dense(8)-Dense(1,sigmoid)",
                "optimizer": "adam",
                "loss": "binary_crossentropy",
                "cv_strategy": "StratifiedKFold(n_splits=6)",
                "early_stopping_patience": 50,
            }
        )
        mlflow.log_metrics(TRAINING_METRICS)

        mlflow.pyfunc.log_model(
            artifact_path="heart_disease_risk_model",
            python_model=HeartDiseaseRiskModel(),
            artifacts={
                "model": config.MODEL_PATH,
                "encoder": config.ENCODER_PATH,
            },
            registered_model_name="heart-disease-risk-classifier",
        )
        print("Model registered in MLflow Model Registry as "
              "'heart-disease-risk-classifier'.")


if __name__ == "__main__":
    main()
