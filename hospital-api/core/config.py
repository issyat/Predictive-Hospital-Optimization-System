"""
S — Single Responsibility: owns all environment/settings logic.
Changes to config source (env file, vault, etc.) only touch this file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    databricks_host:    str = ""
    databricks_token:   str = ""
    prophet_model_uri:  str = "models/prophet/prophet_model"
    xgboost_model_uri:  str = "models/xgboost/xgboost_model"
    jwt_secret_key:     str = "fallback_secret"
    jwt_algorithm:      str = "HS256"
    jwt_expire_minutes: int = 60


# Single shared instance — import this everywhere instead of reading os.getenv directly
settings = Settings()
