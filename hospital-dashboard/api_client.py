"""
Centralised HTTP client for the Hospital API.
All views call these functions — never raw requests.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL      = os.getenv("API_URL", "http://localhost:8000")
API_PASSWORD = os.getenv("API_PASSWORD", "hospital2024")

TIMEOUT = 15


def get_token(username: str) -> str:
    r = requests.post(
        f"{API_URL}/token",
        json={"username": username, "password": API_PASSWORD},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def health() -> dict:
    r = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def forecast(token: str, target_date: str, days: int) -> dict:
    r = requests.post(
        f"{API_URL}/forecast",
        json={"target_date": target_date, "forecast_days": days},
        headers=auth_headers(token),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def staffing(token: str) -> dict:
    r = requests.get(
        f"{API_URL}/staffing",
        headers=auth_headers(token),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def alert(token: str, vitals: dict) -> dict:
    r = requests.post(
        f"{API_URL}/alerts",
        json=vitals,
        headers=auth_headers(token),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()
