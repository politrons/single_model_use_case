from __future__ import annotations

from typing import Any

import mlflow  # type: ignore # noqa
from mlflow.models import EvaluationResult  # type: ignore # noqa

class EvalModelConfig:
    def evaluate(
        self,
        model_for_evaluate: Any,
        data: Any,
        targets: Any,
        model_type: str | None,
        evaluators: list[str],
        evaluator_config: dict[str, Any],
    ) -> EvaluationResult | None:
        return mlflow.models.evaluate(
            model_for_evaluate,
            data=data,
            targets=targets,
            model_type=model_type,
            evaluators=evaluators,
            evaluator_config=evaluator_config,
        )


build = EvalModelConfig()
