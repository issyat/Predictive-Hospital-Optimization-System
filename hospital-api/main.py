# ── main.py ────────────────────────────────────────────────────
"""
Hospital Predictive Optimization System — FastAPI Gateway
App factory: wires middleware, lifecycle events, and routers together.
All business logic lives in services/; all HTTP contracts live in routers/.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from services.model_registry import model_registry
from routers import auth, health, forecast, staffing, alerts

# ── Authenticate against Databricks before anything else loads
os.environ["DATABRICKS_HOST"]  = settings.databricks_host
os.environ["DATABRICKS_TOKEN"] = settings.databricks_token
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")   # models use Unity Catalog 3-level names

app = FastAPI(
    title       = "Hospital Predictive Optimization API",
    description = "FHIR R4 compliant clinical AI gateway",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Lifecycle: load all MLflow models once at startup
@app.on_event("startup")
async def startup() -> None:
    model_registry.load_all()

# ── Routers — each file owns exactly one resource / pipeline
app.include_router(auth.router)
app.include_router(health.router)
app.include_router(forecast.router)
app.include_router(staffing.router)
app.include_router(alerts.router)
