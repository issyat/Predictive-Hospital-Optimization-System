"""
S — Single Responsibility: all Pydantic request models in one place.
I — Interface Segregation: each model exposes only the fields its endpoint needs.
"""
from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    username: str
    password: str


class ForecastRequest(BaseModel):
    target_date: str = Field(
        ...,
        example     = "2151-09-01",
        description = "Date to start forecasting from (YYYY-MM-DD)",
    )
    forecast_days: int = Field(
        default     = 7,
        ge          = 1,
        le          = 30,
        description = "Number of days to forecast",
    )


class AlertRequest(BaseModel):
    heart_rate_mean:           float = Field(..., example=88.0)
    systolic_bp_mean:          float = Field(..., example=110.0)
    spo2_mean:                 float = Field(..., example=95.0)
    temperature_c_mean:        float = Field(..., example=37.4)
    respiratory_rate_mean:     float = Field(..., example=18.0)
    heart_rate_max:            float = Field(..., example=120.0)
    systolic_bp_min:           float = Field(..., example=85.0)
    spo2_min:                  float = Field(..., example=91.0)
    temperature_c_max:         float = Field(..., example=38.2)
    respiratory_rate_max:      float = Field(..., example=24.0)
    heart_rate_last:           float = Field(..., example=95.0)
    spo2_last:                 float = Field(..., example=94.0)
    creatinine_last:           float = Field(..., example=1.2)
    glucose_last:              float = Field(..., example=120.0)
    hemoglobin_last:           float = Field(..., example=12.0)
    wbc_last:                  float = Field(..., example=8.5)
    lactate_last:              float = Field(..., example=1.5)
    temperature_c_was_missing: float = Field(default=0.0)
    age_at_admission:          float = Field(..., example=65.0)
