"""
Lightweight SQLite logging of every prediction, used to power the
monitoring dashboard. Uses only the standard library (sqlite3) so no extra
infra is required to get basic monitoring working.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app import config

_lock = threading.Lock()


def _ensure_parent_dir() -> None:
    Path(config.LOG_DB_PATH).parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection():
    _ensure_parent_dir()
    conn = sqlite3.connect(config.LOG_DB_PATH, timeout=10)
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with _lock, get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                input_json TEXT NOT NULL,
                risk_probability REAL NOT NULL,
                prediction INTEGER NOT NULL,
                risk_label TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                model_version TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at)"
        )
        conn.commit()


def log_prediction(
    request_id: str,
    input_data: dict,
    risk_probability: float,
    prediction: int,
    risk_label: str,
    latency_ms: float,
    model_version: str,
) -> None:
    with _lock, get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO predictions
            (id, created_at, input_json, risk_probability, prediction, risk_label, latency_ms, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(input_data),
                risk_probability,
                prediction,
                risk_label,
                latency_ms,
                model_version,
            ),
        )
        conn.commit()


def get_stats() -> dict:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]

        if total == 0:
            return {
                "total_predictions": 0,
                "positive_predictions": 0,
                "positive_rate": 0.0,
                "avg_risk_probability": 0.0,
                "avg_latency_ms": 0.0,
                "predictions_last_24h": 0,
                "risk_distribution": {"low": 0, "medium": 0, "high": 0},
                "timeseries": [],
            }

        positive = conn.execute(
            "SELECT COUNT(*) AS c FROM predictions WHERE prediction = 1"
        ).fetchone()["c"]

        avg_prob = conn.execute(
            "SELECT AVG(risk_probability) AS a FROM predictions"
        ).fetchone()["a"]

        avg_latency = conn.execute(
            "SELECT AVG(latency_ms) AS a FROM predictions"
        ).fetchone()["a"]

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        last_24h = conn.execute(
            "SELECT COUNT(*) AS c FROM predictions WHERE created_at >= ?", (cutoff,)
        ).fetchone()["c"]

        distribution = {"low": 0, "medium": 0, "high": 0}
        for row in conn.execute(
            "SELECT risk_label, COUNT(*) AS c FROM predictions GROUP BY risk_label"
        ):
            distribution[row["risk_label"]] = row["c"]

        timeseries_rows = conn.execute(
            """
            SELECT substr(created_at, 1, 13) AS hour_bucket,
                   COUNT(*) AS count,
                   AVG(risk_probability) AS avg_risk
            FROM predictions
            GROUP BY hour_bucket
            ORDER BY hour_bucket DESC
            LIMIT 24
            """
        ).fetchall()
        timeseries = [
            {"hour": r["hour_bucket"], "count": r["count"], "avg_risk": round(r["avg_risk"], 4)}
            for r in reversed(timeseries_rows)
        ]

        return {
            "total_predictions": total,
            "positive_predictions": positive,
            "positive_rate": round(positive / total, 4),
            "avg_risk_probability": round(avg_prob, 4),
            "avg_latency_ms": round(avg_latency, 3),
            "predictions_last_24h": last_24h,
            "risk_distribution": distribution,
            "timeseries": timeseries,
        }


def get_recent_predictions(limit: int = 20) -> list:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, risk_probability, prediction, risk_label, latency_ms
            FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
