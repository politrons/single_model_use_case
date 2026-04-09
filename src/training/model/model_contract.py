from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelContract(ABC):
    @abstractmethod
    def get_model(self, args: dict[str, Any]) -> Any:
        """Build and return the model implementation."""

    @abstractmethod
    def log_model(
        self,
        model: Any,
        model_name: str,
        signature: Any,
        input_example: Any,
        args: dict[str, Any],
    ) -> None:
        """Log/register a trained model."""
