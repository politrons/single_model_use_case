from typing import Any

import inspect  # type: ignore # noqa
import numpy as np  # type: ignore # noqa
from sklearn.pipeline import Pipeline  # type: ignore # noqa
from sklearn.compose import TransformedTargetRegressor  # type: ignore # noqa
from sklearn.experimental import enable_halving_search_cv   # type: ignore # noqa
from sklearn.model_selection import GridSearchCV, HalvingRandomSearchCV  # type: ignore # noqa
import xgboost  # type: ignore # noqa
import lightgbm  # type: ignore # noqa
from sklearn.base import (  # type: ignore # noqa
    BaseEstimator,
)
# TensorFlow is optional; avoid hard import at module load time.
try:  # pragma: no cover - depends on optional dependency
    from databricks_mlops_stack.training.model.algorithms.tensorflow_algo import TFBaseModel  # type: ignore # noqa
except Exception:  # pragma: no cover
    class TFBaseModel:  # type: ignore[no-redef]
        pass
from databricks_mlops_stack.utils.constants.model import (   # type: ignore # noqa
    STEP_MODEL,
    STEP_SAMPLER,
    STEP_TARGET_TRANSFORMER,
)

def get_non_default_pipeline_params(
        pipeline: Pipeline | TransformedTargetRegressor | BaseEstimator,
    ) -> dict[str, Any]:

    non_default = {}

    def get_default_params(base_est: BaseEstimator) -> dict:
        sig = inspect.signature(base_est.__init__)
        return {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

    def is_different(current: Any, default: Any) -> bool:
        if current is None and default is None:
            return False
        if current == default:
            return False
        if isinstance(current, float) and isinstance(default, float):
            if np.isnan(current) and np.isnan(default):
                return False
        return True

    # Recursively walk through any estimator/transformer
    def walk_step(step: Any, step_name: str, current_prefix: str):
        if step is None:
            return

        cls = step.__class__
        defaults = {}

        match step:

            case xgboost.sklearn.XGBClassifier() | xgboost.sklearn.XGBRegressor() as xgb:
                defaults = xgb.__class__().get_params()
            case lightgbm.sklearn.LGBMClassifier() | lightgbm.sklearn.LGBMRegressor() as lig:
                defaults = lig.__class__().get_params()
            case s if "ray" in str(s.__class__).lower():
                for name, step in step.estimator.named_steps.items():
                    walk_step(step, name, f"{name}__")
                return
            case GridSearchCV() | HalvingRandomSearchCV():
                for name, step in step.estimator.named_steps.items():
                    walk_step(step, name, f"{name}__")
                return
            case TFBaseModel():
                pass
            case _:
                defaults = get_default_params(cls)

        current_params = step.get_params(deep=False)

        for key, val in current_params.items():
            if key.startswith('_') or key.endswith('_'):
                continue
            default_val = defaults.get(key)
            if is_different(val, default_val):
                full_key = f"{current_prefix}{key}"
                non_default[full_key] = val

        if step_name in [STEP_MODEL, STEP_SAMPLER, STEP_TARGET_TRANSFORMER]:
            non_default[f"{step_name}"] = step

        if hasattr(step, 'steps'):
            for sub_name, sub_step in step.steps:
                sub_prefix = f"{current_prefix}{sub_name}__"
                walk_step(sub_step, f"{step_name}_{sub_name}", sub_prefix)

    match pipeline:

        case Pipeline():

            for name, step in pipeline.named_steps.items():
                walk_step(step, name, f"{name}__")

        case TransformedTargetRegressor():

            walk_step(pipeline.transformer, STEP_TARGET_TRANSFORMER, f"{STEP_TARGET_TRANSFORMER}__")
            for name, step in pipeline.regressor.named_steps.items():
                walk_step(step, name, f"{name}__")

        case BaseEstimator():
            walk_step(pipeline, STEP_MODEL, f"{STEP_MODEL}__")

        case _:
            raise ValueError("Input must be a scikit-learn Pipeline or equivalent")

    return non_default