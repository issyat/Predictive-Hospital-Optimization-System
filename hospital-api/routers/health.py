from datetime import datetime
from fastapi import APIRouter

from services.model_registry import model_registry

router = APIRouter(tags=["System"])


@router.get("/health")
def health_check():
    """System health check — no authentication required."""
    return {
        "status":        "healthy",
        "timestamp":     datetime.utcnow().isoformat(),
        "models_loaded": {
            "prophet": model_registry.prophet is not None,
            "xgboost": model_registry.xgboost is not None,
        },
        "version": "1.0.0",
    }
