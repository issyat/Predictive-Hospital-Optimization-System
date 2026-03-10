from fastapi import APIRouter, Depends

from core.security import verify_token
from services.staffing_service import StaffingService

router = APIRouter(tags=["Pipeline 2 — Staffing"])


def get_staffing_service() -> StaffingService:
    return StaffingService()


@router.get("/staffing")
def get_staffing_schedule(
    service:  StaffingService = Depends(get_staffing_service),
    username: str             = Depends(verify_token),
):
    """Return the latest optimised staff schedule (FHIR R4 Schedule resource)."""
    return service.predict(username)
