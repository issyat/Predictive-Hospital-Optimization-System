"""
S — Single Responsibility: only Pipeline 2 (staff scheduling) logic lives here.
In production, replace _SAMPLE_SCHEDULE with a real Databricks Delta table read
by injecting a data client — no other file needs to change.
"""
from datetime import datetime
from fastapi import HTTPException

from services.base import PredictionService


class StaffingService(PredictionService):
    """Pipeline 2 — staff schedule from Linear Programming optimisation."""

    # Placeholder data — swap for a Databricks/Delta client call in production
    _SAMPLE_SCHEDULE = [
        {"date": "2151-08-08", "day": "Sunday",    "morning": 3, "afternoon": 3, "night": 2, "total": 8},
        {"date": "2151-08-09", "day": "Monday",    "morning": 4, "afternoon": 5, "night": 3, "total": 12},
        {"date": "2151-08-10", "day": "Tuesday",   "morning": 4, "afternoon": 5, "night": 2, "total": 11},
        {"date": "2151-08-11", "day": "Wednesday", "morning": 4, "afternoon": 5, "night": 2, "total": 11},
        {"date": "2151-08-12", "day": "Thursday",  "morning": 4, "afternoon": 5, "night": 2, "total": 11},
        {"date": "2151-08-13", "day": "Friday",    "morning": 4, "afternoon": 5, "night": 2, "total": 11},
        {"date": "2151-08-14", "day": "Saturday",  "morning": 3, "afternoon": 4, "night": 2, "total": 9},
    ]

    # ── PredictionService contract ────────────────────────────────────

    def is_ready(self) -> bool:
        return True  # No ML model needed; data comes from Delta table

    def predict(self, username: str) -> dict:
        try:
            return {
                "resourceType":    "Schedule",
                "requested_by":    username,
                "generated_at":    datetime.utcnow().isoformat(),
                "optimization":    "Linear Programming (PuLP)",
                "kpi_improvement": "42.5%",
                "schedule":        self._SAMPLE_SCHEDULE,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
