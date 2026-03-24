from typing import Any

import numpy.typing as npt   # type: ignore # noqa
import xgboost as xgb  # type: ignore # noqa
import lightgbm as lgb  # type: ignore # noqa
from sklearn.pipeline import Pipeline  # type: ignore # noqa
from sklearn.metrics import get_scorer  # type: ignore # noqa
from sklearn.base import BaseEstimator  # type: ignore # noqa
from sklearn.model_selection._search import BaseSearchCV   # type: ignore # noqa
import numpy as np  # type: ignore # noqa
from sklearn.metrics import (   # type: ignore # noqa
    mean_squared_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
    explained_variance_score,
)
from databricks_mlops_stack.utils.constants.model import (  # type: ignore # noqa
    CONFIG_EVAL_METRIC,
    CONFIG_SCORING,
    METRIC_AUC,
    METRIC_R2_SCORE,
    METRIC_R2,
)

# -----------------------------------------------------------------------------
# Custom functions
# -----------------------------------------------------------------------------
def rmse_score( # now sickit provides RMSE, in theory this is no longer needed
        y_true: npt.NDArray,
        y_pred: npt.NDArray,
    ) -> npt.NDArray:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def false_positives(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pos_label: Any = 1,
    ) -> int:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    if len(unique_labels) > 2:
        raise ValueError("false_positives is only defined for binary classification.")
    elif len(unique_labels) < 2:
        return 0
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))

    return int(fp)

def false_negatives(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pos_label: Any = 1,
    ) -> int:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    if len(unique_labels) > 2:
        raise ValueError("false_negatives is only defined for binary classification.")
    elif len(unique_labels) < 2:
        return 0
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

    return int(fn) 

# -----------------------------------------------------------------------------
# Private constants and functions
# -----------------------------------------------------------------------------
_BUILTIN_METRICS =  {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
    "roc_auc": roc_auc_score,
    METRIC_R2: r2_score,
    "explained_variance": explained_variance_score,
}

_CUSTOM_METRICS = {
    "rmse": (rmse_score, False),
    "mse": (mean_squared_error, False),
    "mean_squared_error": (mean_squared_error, False),
    "accuracy_score": (accuracy_score, get_scorer('accuracy')._sign > 0), # mapping to the same
    "f1_score": (f1_score, get_scorer('f1')._sign > 0), # mapping to the same
    "precision_score": (precision_score, get_scorer('precision')._sign > 0), # mapping to the same
    "recall_score": (recall_score, get_scorer('recall')._sign > 0), # mapping to the same
    METRIC_AUC: (roc_auc_score, get_scorer("roc_auc")._sign > 0), # mapping to the same
    METRIC_R2_SCORE: (r2_score, get_scorer(METRIC_R2)._sign > 0), # mapping to the same
    'false_positives': (false_positives, False),
    'false_negatives': (false_negatives, False),
}

def _get_all_metrics() -> dict[str, Any]:
    result = {}

    not_repeated = _BUILTIN_METRICS.keys().isdisjoint(_CUSTOM_METRICS.keys())
    not_repeated = not_repeated and _CUSTOM_METRICS.keys().isdisjoint(_BUILTIN_METRICS.keys())
    if not not_repeated:
        raise ValueError("Repeated metric names in builtin and custom")

    for k, v in _BUILTIN_METRICS.items():
        result[k] = {
            'function_name': k,
            'function': v,
            'greater_is_better': get_scorer(k)._sign > 0,
        }

    for k, v in _CUSTOM_METRICS.items():
        result[k] = {
            'function_name': k,
            'function': v[0],
            'greater_is_better': v[1],
        }

    return result

def _handle_scikit_search(
        scikit_search_cv: BaseSearchCV,
    ) -> dict[str, Any]:

    scoring = getattr(scikit_search_cv, CONFIG_SCORING, None)
    if not scoring:
        return SCIKIT_DEFAULT

    match scoring:

        case str() as scoring_str:

            return _non_default_scorer_info_from_metric(scoring_str)

        case list() | tuple() | dict():

            refit_estimator = getattr(scikit_search_cv, "refit")

            match refit_estimator:

                case str() as refit_estimator_str:

                    return _non_default_scorer_info_from_metric(refit_estimator_str)

                case _:
                    raise ValueError(f"Custom '{CONFIG_EVAL_METRIC}/{CONFIG_SCORING}' not supported yet")

        case _:
            raise TypeError(f"'{CONFIG_EVAL_METRIC}/{CONFIG_SCORING}' format not recognised")

def _handle_xgboost_lightgbm(
        estimator: xgb.XGBModel | lgb.LGBMModel,
        use_first_metric: bool | None = True,
    ) -> dict[str, Any] | list[dict[str, Any]]:

    eval_metric = estimator.get_params().get(CONFIG_EVAL_METRIC)
    if not eval_metric:
        return SCIKIT_DEFAULT

    match eval_metric:

        case str() as eval_metric_str:

            return _non_default_scorer_info_from_metric(eval_metric_str)

        case list() | tuple() | dict() as eval_metric_collection:

            if use_first_metric:
                return _non_default_scorer_info_from_metric(eval_metric_collection[0])
            else:
                all_scores_info = []
                for m in eval_metric_collection:
                    all_scores_info.append(_non_default_scorer_info_from_metric(m))
                return all_scores_info

        case _:
            raise TypeError(f"'{CONFIG_EVAL_METRIC}/{CONFIG_SCORING}' format not recognised")

def _non_default_scorer_info_from_metric(metric_name: str) -> dict[str, Any]:
    scorer_info = get_metric_info(metric_name)
    scorer_info.update({'is_default_scorer': False})
    _assert_scorer_info(scorer_info)
    return scorer_info

def _assert_scorer_info(scorer_info: dict[str, Any] | list[dict[str, Any]]) -> None:

    def single_scorer_assertion(single_scorer: dict[str, Any]) -> None:
        assert 'function_name' in single_scorer.keys()
        assert 'function' in single_scorer.keys()
        assert 'greater_is_better' in single_scorer.keys()
        assert 'is_default_scorer' in single_scorer.keys()

    match scorer_info:
        case dict() as single:
            single_scorer_assertion(single)
        case list() as collection_scorers:
            for one_scorer in collection_scorers:
                single_scorer_assertion(one_scorer)
        case _:
            raise TypeError("scorer info with wrong configuration")
    return

# -----------------------------------------------------------------------------
# Public constants and functions
# -----------------------------------------------------------------------------
ALL_METRICS = _get_all_metrics()

# Scikit default estimators use accuracy or R2 → greater is better
SCIKIT_DEFAULT = {
    'function_name': None,
    'function': None,
    'greater_is_better': True,
    'is_default_scorer': True,
}

def get_metric_info(
        function_name: str,
    ) -> dict[str, Any]:
    '''
    Get the metric information.

    @function_name: function name

    @return: a dictionary containing:
        - function_name: str (None if it is default scorer)
        - function: Callable (None  if it is default scorer), actual metric function
        - greater_is_better: bool, function signal
    '''

    function_info = ALL_METRICS.get(function_name.lower().strip(), None)

    if not function_info:
        raise ValueError(f"Function {function_name} not mapped")

    return function_info

def get_scorer_info_from_config(scoring: str | None) -> dict[str, Any]:
    if not scoring:
        return SCIKIT_DEFAULT
    match scoring:

        case str() as scoring_str:

            return _non_default_scorer_info_from_metric(scoring_str)

        case list():
            raise ValueError(f"List for '{CONFIG_EVAL_METRIC}/{CONFIG_SCORING}' not supported yet")
        case _ if callable(scoring):
            raise ValueError(f"Custom '{CONFIG_EVAL_METRIC}/{CONFIG_SCORING}' not supported yet")
        case _:
            raise TypeError(f"'{CONFIG_EVAL_METRIC}/{CONFIG_SCORING}' format not recognised")

def get_scorer_info(
        estimator_or_pipe: BaseEstimator | BaseSearchCV | Pipeline,
        use_first_metric: bool | None = True,
    ) -> dict[str, Any] | list[dict[str, Any]]:
    '''
    Get scorer information.

    @estimator_or_pipe: object to extract the scorer
    @use_first_metric: optional param to be passed when multiples metrics are used - If set to false,
    and the estimator have multiple metrics, a list will be returned

    @return: a dictionary (or list of dictionaries) containing: 
        - function_name: str (None if it is default scorer)
        - function: Callable (None  if it is default scorer), actual metric function
        - greater_is_better: bool, function signal
        - is_default_scorer: bool
    '''

    match estimator_or_pipe:
        
        case Pipeline():
            estimator = estimator_or_pipe.steps[-1][1]
        case _:
            if hasattr(estimator_or_pipe, "is_default_scorer"):
                if estimator_or_pipe.is_default_scorer:
                    return SCIKIT_DEFAULT
                else:
                    return _non_default_scorer_info_from_metric(estimator_or_pipe.metric)
            else:
                estimator = estimator_or_pipe
    
    match estimator:

        case xgb.XGBModel() | lgb.LGBMModel():

            scorer_info = _handle_xgboost_lightgbm(estimator, use_first_metric)  # type: ignore # noqa

        case s if 'tensorflow' in str(s.__class__).lower():
            raise TypeError("Scoring functionality not implemented yet for TensorFlow models")

        case BaseSearchCV():

            scorer_info = _handle_scikit_search(estimator)

        case BaseEstimator():

            scorer_info = SCIKIT_DEFAULT

        case _:
            raise TypeError("estimator not recognised")
        
    _assert_scorer_info(scorer_info)

    return scorer_info
