"""
S — Single Responsibility: only Pipeline 1 (admission forecasting) logic lives here.
O — Extending forecast behaviour (new freq, new model) requires no changes to other services.
"""
import pandas as pd
from fastapi import HTTPException

from schemas.requests import ForecastRequest
from services.base import PredictionService
from services.model_registry import ModelRegistry


class ForecastService(PredictionService):
    """Pipeline 1 — admission forecasting with Prophet."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    # ── PredictionService contract ────────────────────────────────────

    def is_ready(self) -> bool:
        return self._registry.prophet is not None

    def predict(self, request: ForecastRequest, username: str) -> dict:
        if not self.is_ready():
            raise HTTPException(status_code=503, detail="Forecasting model not available")
        try:
            return self._build_fhir_bundle(request, username)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Private helpers ───────────────────────────────────────────────

    def _build_fhir_bundle(self, request: ForecastRequest, username: str) -> dict:
        target_date = pd.to_datetime(request.target_date)
        model       = self._registry.prophet

        future_df   = model.make_future_dataframe(periods=request.forecast_days + 30, freq="D")
        forecast_df = model.predict(future_df)

        window = forecast_df[
            (forecast_df["ds"] >= target_date) &
            (forecast_df["ds"] <  target_date + pd.Timedelta(days=request.forecast_days))
        ][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

        if window.empty:
            raise HTTPException(
                status_code=400,
                detail=f"Date {request.target_date} is outside the model's forecast range",
            )

        return {
            "resourceType":  "Bundle",
            "type":          "collection",
            "requested_by":  username,
            "forecast_days": request.forecast_days,
            "entry":         [self._to_observation(row) for _, row in window.iterrows()],
        }

    @staticmethod
    def _to_observation(row) -> dict:
        """Convert a single forecast row to a FHIR R4 Observation resource."""
        return {
            "resourceType": "Observation",
            "status":       "preliminary",
            "code": {
                "coding": [{
                    "system":  "http://loinc.org",
                    "code":    "75325-1",
                    "display": "Hospital admission count forecast",
                }]
            },
            "effectiveDateTime": str(row["ds"])[:10],
            "dayOfWeek":         row["ds"].day_name(),
            "valueQuantity": {
                "value":  max(0, round(row["yhat"])),
                "unit":   "admissions",
                "system": "http://unitsofmeasure.org",
            },
            "referenceRange": [{
                "low":  {"value": max(0, round(row["yhat_lower"]))},
                "high": {"value": max(0, round(row["yhat_upper"]))},
            }],
        }
