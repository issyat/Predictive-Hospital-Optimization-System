"""
S — Single Responsibility: the only place that talks to MLflow.
D — Dependency Inversion: services receive this registry rather than
    calling mlflow directly, so they depend on the abstraction (registry)
    not the concrete tracking library.
"""
import mlflow
import mlflow.prophet
import mlflow.xgboost

from core.config import settings


class ModelRegistry:
    """Loads and caches all MLflow models at startup."""

    def __init__(self) -> None:
        self._prophet   = None
        self._xgboost   = None
        self._explainer = None   # SHAP TreeExplainer — built lazily on first use

    def load_all(self) -> None:
        print("Loading models from MLflow Registry...")
        print(f"   Prophet URI : {settings.prophet_model_uri}")
        print(f"   XGBoost URI : {settings.xgboost_model_uri}")
        self._prophet = self._load(settings.prophet_model_uri, mlflow.prophet.load_model, "Prophet")
        self._xgboost = self._load_xgboost(settings.xgboost_model_uri)
        print("Models ready")

    @staticmethod
    def _load(uri: str, loader, name: str):
        try:
            model = loader(uri)
            print(f"   ✅ {name} loaded")
            return model
        except Exception as exc:
            print(f"   ⚠️  {name} not loaded: {exc}")
            return None

    @staticmethod
    def _load_xgboost(model_dir: str):
        """Load the raw XGBoost Booster directly from the .xgb file.
        Uses xgb.Booster instead of XGBClassifier to avoid any sklearn dependency."""
        import os
        import xgboost as xgb
        xgb_file = os.path.join(model_dir, "model.xgb")
        try:
            booster = xgb.Booster()
            booster.load_model(xgb_file)
            print("   ✅ XGBoost loaded")
            return booster
        except Exception as exc:
            print(f"   ⚠️  XGBoost not loaded: {exc}")
            return None

    @property
    def prophet(self):
        return self._prophet

    @property
    def xgboost(self):
        return self._xgboost

    @property
    def xgboost_explainer(self):
        """Lazily build and cache a SHAP TreeExplainer for the XGBoost Booster."""
        if self._xgboost is not None and self._explainer is None:
            try:
                import shap
                self._explainer = shap.TreeExplainer(self._xgboost)
                print("   ✅ SHAP explainer ready")
            except Exception as exc:
                print(f"   ⚠️  SHAP explainer not built: {exc}")
        return self._explainer


# Module-level singleton — imported by routers via get_*_service() factories
model_registry = ModelRegistry()
