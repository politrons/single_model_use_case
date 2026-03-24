from typing import Any

from sklearn.base import BaseEstimator  # type: ignore # noqa
from sklearn.model_selection._search import BaseSearchCV  # type: ignore # noqa
from sklearn.model_selection._split import BaseCrossValidator  # type: ignore # noqa
from sklearn.experimental import enable_halving_search_cv   # type: ignore # noqa
from sklearn.model_selection import GridSearchCV, HalvingRandomSearchCV  # type: ignore # noqa
from scipy import stats as sp  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.model import (  # type: ignore # noqa
    SCIKIT_GRID,
    SCIKIT_HALVING_RANDOM,
    STEP_MODEL,
)

# -----------------------------------------------------------------------------
# Catalogs
# -----------------------------------------------------------------------------
AVAILABLE_SCIPY_SAMPLING = {
    "randint": sp.randint,
    "uniform": sp.uniform,
    "loguniform": sp.loguniform,
}


def convert_user_dict_to_scipy_dict(user_config: dict[str, Any]) -> dict[str, Any]:
    scipy_space = {}
    for param_name, (func_name, *args) in user_config.items():
        if func_name not in AVAILABLE_SCIPY_SAMPLING:
            raise ValueError(f"Unsupported SciPy sampling: {func_name}")
        func = AVAILABLE_SCIPY_SAMPLING[func_name]
        scipy_space[param_name] = func(*args)
    return scipy_space


def scikit_hyperparam_search(
        search_strategy: str,
        pipe: BaseEstimator,
        all_spaces: dict[str, Any],
        cv_splitter: BaseCrossValidator,
        random_state: int,
        parallelism: int | None,
        factor: int,
        scoring: str | None,  # when adding support for custom/multiple metrics, change here
    ) -> BaseSearchCV:

    search = None

    if search_strategy == SCIKIT_GRID:
        search = GridSearchCV(
            estimator=pipe,
            param_grid={f"{STEP_MODEL}__" + k: v for k, v in all_spaces.items()},
            cv=cv_splitter,
            refit=True,
            n_jobs=parallelism,
            scoring=scoring,
        )
    elif search_strategy == SCIKIT_HALVING_RANDOM:
        converted_params = convert_user_dict_to_scipy_dict(all_spaces)
        search = HalvingRandomSearchCV(
            estimator=pipe,
            param_distributions={f"{STEP_MODEL}__" + k: v for k, v in converted_params.items()},
            cv=cv_splitter,
            refit=True,
            n_jobs=parallelism,
            random_state=random_state,
            factor=factor,
            scoring=scoring,
        )
    else:
        raise ValueError(f"Search strategy {search_strategy} not implemented.")

    return search