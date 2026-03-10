"""
S — Single Responsibility: only Pipeline 3 (patient risk alerts) logic lives here.
Threshold and feature list are module-level constants — easy to tune without
touching any other service.
"""
import pandas as pd
from datetime import datetime
from fastapi import HTTPException

from schemas.requests import AlertRequest
from services.base import PredictionService
from services.model_registry import ModelRegistry

FEATURE_COLUMNS: list[str] = [
    "heart_rate_mean", "systolic_bp_mean", "spo2_mean",
    "temperature_c_mean", "respiratory_rate_mean",
    "heart_rate_max", "systolic_bp_min", "spo2_min",
    "temperature_c_max", "respiratory_rate_max",
    "heart_rate_last", "spo2_last",
    "creatinine_last", "glucose_last", "hemoglobin_last",
    "wbc_last", "lactate_last",
    "temperature_c_was_missing", "age_at_admission",
]

# Human-readable labels shown in the dashboard SHAP chart
FEATURE_LABELS: dict[str, str] = {
    "heart_rate_mean":           "Heart Rate — mean (bpm)",
    "systolic_bp_mean":          "Systolic BP — mean (mmHg)",
    "spo2_mean":                 "SpO₂ — mean (%)",
    "temperature_c_mean":        "Temperature — mean (°C)",
    "respiratory_rate_mean":     "Respiratory Rate — mean (br/min)",
    "heart_rate_max":            "Heart Rate — max (bpm)",
    "systolic_bp_min":           "Systolic BP — min (mmHg)",
    "spo2_min":                  "SpO₂ — min (%)",
    "temperature_c_max":         "Temperature — max (°C)",
    "respiratory_rate_max":      "Respiratory Rate — max (br/min)",
    "heart_rate_last":           "Heart Rate — last (bpm)",
    "spo2_last":                 "SpO₂ — last (%)",
    "creatinine_last":           "Creatinine — last (mg/dL)",
    "glucose_last":              "Glucose — last (mg/dL)",
    "hemoglobin_last":           "Hemoglobin — last (g/dL)",
    "wbc_last":                  "WBC — last (×10³/µL)",
    "lactate_last":              "Lactate — last (mmol/L)",
    "temperature_c_was_missing": "Temperature data missing",
    "age_at_admission":          "Age at admission (years)",
}

RISK_THRESHOLD: float = 0.70


class AlertService(PredictionService):
    """Pipeline 3 — patient complication risk with XGBoost."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    # ── PredictionService contract ────────────────────────────────────

    def is_ready(self) -> bool:
        return self._registry.xgboost is not None

    def predict(self, request: AlertRequest, username: str) -> dict:
        if not self.is_ready():
            raise HTTPException(status_code=503, detail="Alert model not available")
        try:
            import xgboost as xgb
            features_df = self._build_features_df(request)
            features_dm = xgb.DMatrix(features_df)
            risk_score  = float(self._registry.xgboost.predict(features_dm)[0])
            explanation = self._get_shap_explanation(features_df)
            return self._to_risk_assessment(risk_score, username, explanation)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Private helpers ───────────────────────────────────────────────

    @staticmethod
    def _build_features_df(request: AlertRequest) -> pd.DataFrame:
        return pd.DataFrame([{col: getattr(request, col) for col in FEATURE_COLUMNS}])

    def _get_shap_explanation(self, features_df: pd.DataFrame) -> list[dict]:
        """Return top-8 SHAP feature contributions sorted by absolute impact."""
        explainer = self._registry.xgboost_explainer
        if explainer is None:
            return []
        try:
            shap_values = explainer.shap_values(features_df)
            # Binary classifiers may return list-of-arrays; take positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            shap_row = shap_values[0]
            contributions = [
                {
                    "feature": col,
                    "label":   FEATURE_LABELS.get(col, col),
                    "value":   round(float(features_df.iloc[0][col]), 3),
                    "shap":    round(float(sv), 4),
                }
                for col, sv in zip(FEATURE_COLUMNS, shap_row)
            ]
            return sorted(contributions, key=lambda x: abs(x["shap"]), reverse=True)[:8]
        except Exception:
            return []

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.85:
            return "HIGH"
        if score >= 0.50:
            return "MODERATE"
        return "LOW"

    @staticmethod
    def _to_risk_assessment(risk_score: float, username: str, explanation: list) -> dict:
        """Format result as a FHIR R4 RiskAssessment resource."""
        alert_triggered = risk_score >= RISK_THRESHOLD
        risk_level      = AlertService._risk_level(risk_score)

        return {
            "resourceType": "RiskAssessment",
            "status":       "final",
            "requested_by": username,
            "timestamp":    datetime.utcnow().isoformat(),
            "prediction": [{
                "outcome": {
                    "coding": [{
                        "system":  "http://snomed.info/sct",
                        "code":    "397945004",
                        "display": "Unexpected hospital death",
                    }]
                },
                "probabilityDecimal": round(risk_score, 4),
                "riskLevel":          risk_level,
                "alertTriggered":     alert_triggered,
                "threshold":          RISK_THRESHOLD,
            }],
            "explanation": explanation,
            "note": [{
                "text": (
                    f"Patient risk score: {round(risk_score * 100, 1)}%. "
                    f"Alert {'TRIGGERED' if alert_triggered else 'not triggered'}. "
                    f"Threshold: {RISK_THRESHOLD * 100}%."
                )
            }],
        }
