"""
Pydantic request/response models for the Heart Disease Risk API.

The field set and categorical options below mirror exactly what the model
was trained on (see /scripts/register_model.py and the original training
notebook). Do not rename or reorder fields without retraining/re-exporting
the encoder.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class Sex(str, Enum):
    male = "Male"
    female = "Female"


class ChestPainType(str, Enum):
    asymptomatic = "Asymptomatic"
    non_anginal_pain = "Non-Anginal Pain"
    atypical_angina = "Atypical Angina"
    typical_angina = "Typical Angina"


class SmokerStatus(str, Enum):
    never = "Never"
    current = "Current"
    former = "Former"


class PatientData(BaseModel):
    """A single patient's clinical + lifestyle record."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 54,
                "sex": "Male",
                "resting_bp_systolic": 132,
                "resting_bp_diastolic": 84,
                "cholesterol_total": 221,
                "hdl": 45,
                "ldl": 130,
                "triglycerides": 150,
                "fasting_blood_sugar": 110,
                "hba1c": 5.8,
                "bmi": 27.4,
                "resting_heart_rate": 72,
                "max_heart_rate_achieved": 158,
                "chest_pain_type": "Atypical Angina",
                "exercise_induced_angina": False,
                "st_depression": 1.2,
                "family_history": True,
                "smoker_status": "Former",
                "alcohol_units_per_week": 3.5,
                "exercise_minutes_per_week": 90,
                "sleep_hours": 6.5,
                "stress_score": 6.2,
                "wearable_owner": True,
                "daily_steps": 6500,
                "diet_quality_score": 58.0,
            }
        }
    )

    age: int = Field(..., ge=0, le=120)
    sex: Sex
    resting_bp_systolic: int = Field(..., ge=60, le=260)
    resting_bp_diastolic: int = Field(..., ge=30, le=180)
    cholesterol_total: int = Field(..., ge=50, le=700)
    hdl: int = Field(..., ge=5, le=200)
    ldl: int = Field(..., ge=5, le=500)
    triglycerides: int = Field(..., ge=10, le=1000)
    fasting_blood_sugar: int = Field(..., ge=40, le=500)
    hba1c: float = Field(..., ge=2.0, le=18.0)
    bmi: float = Field(..., ge=10.0, le=70.0)
    resting_heart_rate: int = Field(..., ge=30, le=220)
    max_heart_rate_achieved: int = Field(..., ge=60, le=250)
    chest_pain_type: ChestPainType
    exercise_induced_angina: bool
    st_depression: float = Field(..., ge=0.0, le=10.0)
    family_history: bool
    smoker_status: SmokerStatus
    alcohol_units_per_week: float = Field(..., ge=0.0, le=100.0)
    exercise_minutes_per_week: int = Field(..., ge=0, le=2000)
    sleep_hours: float = Field(..., ge=0.0, le=24.0)
    stress_score: float = Field(..., ge=0.0, le=10.0)
    wearable_owner: bool
    daily_steps: int = Field(..., ge=0, le=100000)
    diet_quality_score: float = Field(..., ge=0.0, le=100.0)


class RiskLabel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = no heart disease, 1 = heart disease")
    risk_probability: float = Field(..., description="Model output probability, 0-1")
    risk_label: RiskLabel
    model_version: str
    latency_ms: float
    request_id: str
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    encoder_loaded: bool
    model_version: str


class StatsResponse(BaseModel):
    total_predictions: int
    positive_predictions: int
    positive_rate: float
    avg_risk_probability: float
    avg_latency_ms: float
    predictions_last_24h: int
    risk_distribution: dict
    timeseries: list
