"""
O — Open/Closed: new pipeline services extend this base without modifying it.
L — Liskov Substitution: all concrete services are drop-in replacements here.
"""
from abc import ABC, abstractmethod


class PredictionService(ABC):
    """Abstract contract every pipeline service must fulfill."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True when the service has everything it needs to handle requests."""
        ...

    @abstractmethod
    def predict(self, *args, **kwargs) -> dict:
        """Run the pipeline and return a FHIR-formatted response dict."""
        ...
