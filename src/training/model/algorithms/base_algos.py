from typing import Any
import logging

from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore # noqa
from sklearn.base import BaseEstimator  # type: ignore # noqa
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # type: ignore # noqa
from sklearn.linear_model import RidgeClassifier, Ridge, LogisticRegression  # type: ignore # noqa
from xgboost import XGBClassifier, XGBRegressor  # type: ignore # noqa

# TensorFlow is optional in some environments (for example unit-test sandboxes).
# We keep import-time behavior resilient and fail only when the TensorFlow algorithm
# is explicitly selected.
try:  # pragma: no cover - branch depends on optional dependency presence
    from databricks_mlops_stack.training.model.algorithms.tensorflow_algo import TFClassifier, TFRegressor  # type: ignore # noqa
    _TENSORFLOW_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    TFClassifier = None  # type: ignore[assignment]
    TFRegressor = None  # type: ignore[assignment]
    _TENSORFLOW_IMPORT_ERROR = exc
from databricks_mlops_stack.training.model.algorithms.neuralprophet_algo import NeuralProphetRegressor  # type: ignore # noqa
from databricks_mlops_stack.utils.mlops_utils import ParserUtils  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.model import (  # type: ignore # noqa
    CONFIG_SECTION_TARGET_TRANSFORMER,
    CONFIG_SECTION_SAMPLING,
    CONFIG_MODEL_CLASS,
    CONFIG_TASK,
    ALGO_XGBOOST,
    ALGO_LIGHTGBM,
    ALGO_SCIKIT_RANDOMFOREST,
    ALGO_SCIKIT_RIDGE,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    CONFIG_SECTION_MODEL,
    ALGO_SCIKIT_LINEAR,
    CONFIG_SECTION_EARLY_STOPPING,
    ALGO_TENSORFLOW,
    CONFIG_SECTION_FEATURES_TRANSFORMERS,
    CONFIG_TRANSFORMER_CLASS,
    CONFIG_SECTION_HYPERPARAM_SEARCH,
    CONFIG_SECTION_DISCARDED_FEATURES,
    ALGO_NEURALPROPHET,
)
from databricks_mlops_stack.utils.constants.core import (  # type: ignore # noqa
    CONFIG_RANDOM_STATE,
    CONFIG_FEATURE_COLUMNS,
    CONFIG_TEMPORAL_COLUMN_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOG = logging.getLogger("framework.training.model.algorithms.base_algos")

# -----------------------------------------------------------------------------
# Catalogs
# -----------------------------------------------------------------------------
AVAILABLE_MODELS = {
    ALGO_SCIKIT_RANDOMFOREST: {
        TASK_CLASSIFICATION: RandomForestClassifier,
        TASK_REGRESSION: RandomForestRegressor,
    },
    ALGO_SCIKIT_RIDGE: {
        TASK_CLASSIFICATION: RidgeClassifier,
        TASK_REGRESSION: Ridge,
    },
    ALGO_XGBOOST: {
        TASK_CLASSIFICATION: XGBClassifier,
        TASK_REGRESSION: XGBRegressor,
    },
    ALGO_LIGHTGBM: {
        TASK_CLASSIFICATION: LGBMClassifier,
        TASK_REGRESSION: LGBMRegressor,
    },
    ALGO_SCIKIT_LINEAR: {
        TASK_CLASSIFICATION: LogisticRegression,
    },
    ALGO_TENSORFLOW: {
        TASK_CLASSIFICATION: TFClassifier,
        TASK_REGRESSION: TFRegressor,
    },
    ALGO_NEURALPROPHET: {
        TASK_REGRESSION: NeuralProphetRegressor,
    },
}

def _get_model_class(
        random_state: int,
        config: dict[str, Any],
    ) -> tuple[BaseEstimator, str]:

    ## validation for random state
    if not isinstance(random_state, int):
        raise ValueError("random_state must be an integer")

    ## validation block: sampling and target_transformer
    has_sampling = config.get(CONFIG_SECTION_SAMPLING) is not None
    has_targettransformer = config.get(CONFIG_SECTION_TARGET_TRANSFORMER) is not None
    model_config = config[CONFIG_SECTION_MODEL]
    problem_task = model_config[CONFIG_TASK].lower().strip()

    # both are not used at the sanme time
    if has_sampling and has_targettransformer:
        raise ValueError(f"Cannot specify both '{CONFIG_SECTION_SAMPLING}' and '{CONFIG_SECTION_TARGET_TRANSFORMER}'")

    # sampling only when allowed
    task_allows_sampling = problem_task in [TASK_CLASSIFICATION]
    if has_sampling and not task_allows_sampling:
        raise ValueError(f"Cannot specify '{CONFIG_SECTION_SAMPLING}' with model task '{problem_task}'")

    # target transformer only when allowed
    task_allows_targettransformer = problem_task in [TASK_REGRESSION]
    if has_targettransformer and not task_allows_targettransformer:
        raise ValueError(f"Cannot specify '{CONFIG_SECTION_TARGET_TRANSFORMER}' with model task '{problem_task}'")

    ## validate model class and return it
    model_base_class = model_config[CONFIG_MODEL_CLASS].lower().strip()

    if model_base_class not in AVAILABLE_MODELS or problem_task not in AVAILABLE_MODELS[model_base_class]:
        raise ValueError(f"Unsupported model '{model_base_class}' for task '{problem_task}'")

    model_class = AVAILABLE_MODELS[model_base_class][problem_task]
    if model_class is None and model_base_class == ALGO_TENSORFLOW:
        raise ImportError(
            "TensorFlow dependencies are not installed. Install 'tensorflow' to use ALGO_TENSORFLOW."
        ) from _TENSORFLOW_IMPORT_ERROR

    return model_class, model_base_class
    
def _get_base_params(
        random_state: int,
        model_config: dict[str, Any],
    ) -> dict[str, Any]:

    # Common defaults propagated when supported by the estimator
    base_params = {CONFIG_RANDOM_STATE: random_state}
    base_params.update(model_config.get("params", {}))
    base_params = ParserUtils.parse_dictionary_params(base_params)

    return base_params

def _get_early_stopping_params(
        maybe_eartly_stop_config: dict[str, Any] | None,
    ) -> dict[str, Any]:

    early_params = maybe_eartly_stop_config or {}
    # in future, modifications will be applied here for more advanced early stopping strategies

    return early_params

def build_configured_estimator(
        config: dict[str, Any],
        random_state: int,
    ) -> BaseEstimator:

    model_class, model_base_class = _get_model_class(random_state, config)

    base_params = _get_base_params(random_state, config[CONFIG_SECTION_MODEL])

    maybe_early_stop_config: dict[str, Any] | None = config.get(CONFIG_SECTION_EARLY_STOPPING)
    base_params.update(_get_early_stopping_params(maybe_early_stop_config))

    # this will be used for algos with custom API
    # for now, only tensorflow
    if model_base_class in [ALGO_TENSORFLOW, ALGO_NEURALPROPHET]:
        LOG.info("non scikit model base class detected")

        extra_params = base_params.copy()
        extra_params['has_hyper_search'] = True if CONFIG_SECTION_HYPERPARAM_SEARCH in config else False
        extra_params['all_feature_transformers'] = []
        all_transformers = config.get(CONFIG_SECTION_FEATURES_TRANSFORMERS, [])
        for feature_transformer in all_transformers:
            extra_params['all_feature_transformers'].append(feature_transformer[CONFIG_TRANSFORMER_CLASS])

        number_actually_used_features = len([
            feat for feat
            in config[CONFIG_FEATURE_COLUMNS]
            if feat not in config.get(CONFIG_SECTION_DISCARDED_FEATURES, [])
        ])
        extra_params['number_used_features'] = number_actually_used_features

        extra_params[CONFIG_TEMPORAL_COLUMN_NAME] = config.get(CONFIG_TEMPORAL_COLUMN_NAME)
        extra_params[CONFIG_FEATURE_COLUMNS] = config.get(CONFIG_FEATURE_COLUMNS)

        model = model_class(
            config=config[CONFIG_SECTION_MODEL],
            random_state=random_state,
            params=base_params,
            extra_params=extra_params,
        )

    # this is used for models with scikit-learn friedly API
    else:
        LOG.info("scikit model base class detected")
        model = model_class(**base_params)

    return model