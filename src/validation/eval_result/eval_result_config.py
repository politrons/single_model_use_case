import logging
import operator
from typing import Any

from mlflow.models import EvaluationResult   # type: ignore # noqa

# ----------------------------- logging -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOG = logging.getLogger("framework.validate.eval_result")

# Supported comparison operators
_OPS: dict[str, Any] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}

class EvalResultConfig:

    def eval_result(self, result: EvaluationResult | None, rules: dict[str, dict[str, float | str]]) -> None:
        """Assert scalar metrics in an MLflow EvaluationResult.

        Parameters
        ----------
        result : mlflow.models.EvaluationResult | None
            The object returned by mlflow.evaluate(...). If None, validation fails.
        rules : dict[str, dict]
            Example: {
                    "accuracy": {"op": ">=", "value": 0.90},
                    "log_loss": {"op": "<=", "value": 0.5}
                    }

        Raises
        ------
        AssertionError
            If any metric is missing or does not satisfy the rule.
        ValueError
            If an unknown operator is used.
        """
        if result is None:
            raise AssertionError("Evaluation result is None. Model evaluation did not produce metrics.")

        if not hasattr(result, "metrics"):
            raise AssertionError("Given 'result' has no 'metrics' attribute. Did you pass EvaluationResult?")

        # Check missing metrics first
        missing = [m for m in rules.keys() if m not in result.metrics]
        if missing:
            available = ", ".join(sorted(result.metrics.keys()))
            raise AssertionError(f"Missing metric(s): {missing}. Available: {available}")

        failures: list[str] = []
        for name, rule in rules.items():
            op = str(rule.get("op", ">=")).strip()
            if op not in _OPS:
                error_message=f"Unknown operator '{op}' for metric '{name}'. Use one of {sorted(_OPS)}"
                LOG.error(error_message)
                raise ValueError(error_message)
            try:
                expected = float(rule["value"])  # type: ignore[index]
            except Exception as e:  # pragma: no cover
                error_message =f"Rule for '{name}' must include numeric 'value' (error: {e})"
                LOG.error(error_message)
                raise ValueError(error_message)

            actual_raw = result.metrics[name]
            try:
                actual = float(actual_raw)
            except Exception:
                error_message =f"Metric '{name}' is not numeric: {actual_raw!r}"
                LOG.error(error_message)
                raise AssertionError(error_message)

            if not _OPS[op](actual, expected):
                failures.append(f"{name}: {actual} {op} {expected}  (FAIL)")

        if failures:
            raise AssertionError("\n".join(failures))

build = EvalResultConfig()
