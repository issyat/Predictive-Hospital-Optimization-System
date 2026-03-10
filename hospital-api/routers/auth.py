from fastapi import APIRouter, HTTPException, status

from core.config import settings
from core.security import create_token
from schemas.requests import TokenRequest

router = APIRouter(tags=["Auth"])


@router.post("/token")
def get_token(request: TokenRequest):
    """
    Obtain a JWT token.
    Pass it as `Authorization: Bearer <token>` on every protected endpoint.
    """
    # Demo auth — replace with a real user store in production
    if request.password != "hospital2024":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_token(request.username)
    return {
        "access_token": token,
        "token_type":   "bearer",
        "expires_in":   settings.jwt_expire_minutes * 60,
    }
