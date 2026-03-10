from fastapi import APIRouter, Depends

from core.security import verify_token
from schemas.requests import AlertRequest
from services.alert_service import AlertService
from services.model_registry import model_registry

router = APIRouter(tags=["Pipeline 3 — Alerts"])


def get_alert_service() -> AlertService:
    return AlertService(model_registry)


@router.post("/alerts")
def predict_complication_risk(
    request:  AlertRequest,
    service:  AlertService = Depends(get_alert_service),
    username: str          = Depends(verify_token),
):
    """Predict complication risk for a single patient. Returns FHIR R4 RiskAssessment."""
    return service.predict(request, username)
