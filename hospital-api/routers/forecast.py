from fastapi import APIRouter, Depends

from core.security import verify_token
from schemas.requests import ForecastRequest
from services.forecast_service import ForecastService
from services.model_registry import model_registry

router = APIRouter(tags=["Pipeline 1 — Forecasting"])


# D — Dependency Inversion: router depends on ForecastService abstraction,
# not on mlflow or pandas directly.
def get_forecast_service() -> ForecastService:
    return ForecastService(model_registry)


@router.post("/forecast")
def forecast_admissions(
    request:  ForecastRequest,
    service:  ForecastService = Depends(get_forecast_service),
    username: str             = Depends(verify_token),
):
    """Predict hospital admission counts for upcoming days. Returns FHIR R4 Bundle."""
    return service.predict(request, username)
