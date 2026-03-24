from typing import Any

import pandas as pd  # type: ignore # noqa
import numpy as np  # type: ignore # noqa
from sklearn.base import (  # type: ignore # noqa
    BaseEstimator,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.metrics import (   # type: ignore # noqa
    accuracy_score,
    r2_score,
)
from sklearn.model_selection._split import BaseCrossValidator  # type: ignore # noqa
from ray import tune, air  # type: ignore # noqa
from ray.tune.schedulers import (  # type: ignore # noqa
    ASHAScheduler,
    FIFOScheduler,
    MedianStoppingRule,
)
from ray.tune.search.optuna import OptunaSearch  # type: ignore # noqa
from ray.tune.search import BasicVariantGenerator  # type: ignore # noqa
from ray.train import report  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.model import (  # type: ignore # noqa
    STEP_MODEL,
)

# -----------------------------------------------------------------------------
# Catalogs
# -----------------------------------------------------------------------------
AVAILABLE_RAY_SCHEDULERS = {
    "ashascheduler": ASHAScheduler(),
    "fifoscheduler": FIFOScheduler(),
    "medianstoppingrule": MedianStoppingRule(),
}

AVAILABLE_RAY_SAMPLING = {
    "choice": tune.choice,
    "randint": tune.randint,
    "qrandint": tune.qrandint,
    "uniform": tune.uniform,
    "quniform": tune.quniform,
    "loguniform": tune.loguniform,
    "qloguniform": tune.qloguniform,
}

AVAILABLE_RAY_SEARCH_ALGOS = {
    "optuna": OptunaSearch(),
}

class RaySearchCV(BaseEstimator):  # type: ignore # noqa
    def __init__(
        self,
        estimator: BaseEstimator,
        user_param_space: dict[str, Any],
        cv_splitter: BaseCrossValidator,
        random_state: int,
        num_samples: int,
        scorer_info: dict[str, Any],
        scheduler_name: str | None = None,
        search_algo_name: str | None = None,
    ):
        self.estimator = estimator
        self.cv_splitter = cv_splitter
        self.random_state = random_state
        self.num_samples = num_samples
        self.scorer_info = scorer_info
        self.scheduler_name = scheduler_name
        self.user_param_space = user_param_space
        self.search_algo_name = search_algo_name

        self.best_params_ = None
        self.best_estimator_: BaseEstimator = BaseEstimator()
        self.best_score_ = None
        self.finished_trials = None

        if self.scheduler_name:
            self.scheduler = AVAILABLE_RAY_SCHEDULERS.get(self.scheduler_name.lower().strip(), None)
            if not self.scheduler:
                raise ValueError(f"Unsupported Ray scheduler: {self.scheduler_name}")
        else:
            self.scheduler = FIFOScheduler() # ray default, setting for later reporting

        if self.search_algo_name:
            self.search_algo = AVAILABLE_RAY_SEARCH_ALGOS.get(self.search_algo_name.lower().strip(), None)
            if not self.search_algo:
                raise ValueError(f"Unsupported Ray algo searcher: {self.search_algo_name}")
        else:
            self.search_algo = BasicVariantGenerator()

        ray_param_space = {}
        for param_name, (func_name, *args) in self.user_param_space.items():
            func = AVAILABLE_RAY_SAMPLING.get(func_name.lower().strip(), None)
            if not func:
                raise ValueError(f"Unsupported Ray sampling: {func_name}")
            ray_param_space[param_name] = func(*args)
        self.param_space = {f"{STEP_MODEL}__" + k: v for k, v in ray_param_space.items()}

        self.mode = "max" if self.scorer_info['greater_is_better'] else "min"
        self.is_default_scorer = self.scorer_info['is_default_scorer']
        if self.is_default_scorer:
            self.metric = 'score'
        else:
            self.metric = self.scorer_info['function_name']
            self.metric_function = self.scorer_info['function']

    def _objective(
            self,
            config: dict[str, Any],
            X: pd.DataFrame,
            y: pd.Series,
        ) -> None:
        scores = []

        for train_idx, val_idx in self.cv_splitter.split(X, y):

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            match y:
                case np.ndarray():
                    y_train, y_val = y[train_idx], y[val_idx]
                case pd.Series() | pd.DataFrame():
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                case _:
                    raise TypeError("target must be nparray or pandas series or dataframe")

            model = clone(self.estimator)
            model.set_params(**config)

            model.fit(X_train, y_train)

            if self.is_default_scorer:
                this_score = model.score(X_val, y_val)
            else:
                this_preds = model.predict(X_val)
                this_score = self.metric_function(y_val, this_preds)

            scores.append(this_score)

        mean_score = np.mean(scores)

        report({self.metric: mean_score})

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
        ) -> None:

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(self._objective, X=X, y=y),
                resources={"cpu": 1},
            ),
            param_space=self.param_space,
            tune_config=tune.TuneConfig(
                metric=self.metric,
                mode=self.mode,
                search_alg=self.search_algo,
                scheduler=self.scheduler,
                num_samples=self.num_samples,
                trial_name_creator=lambda t: f'f{t.trial_id}',
                trial_dirname_creator=lambda t: f'f{t.trial_id}',
            ),
            run_config=air.config.RunConfig(
                name='ray_exp',
            ),
        )

        results = tuner.fit()

        if results.num_terminated == 0:
            raise RuntimeError("No valid trial results found; check your objective function.")

        best_result = results.get_best_result(metric=self.metric, mode=self.mode)

        self.best_params_ = best_result.config
        self.best_score_ = best_result.metrics[self.metric]

        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**self.best_params_)  # type: ignore
        self.best_estimator_.fit(X, y)  # type: ignore

        self.finished_trials = len(  # type: ignore
            results.get_dataframe(filter_metric=self.metric, filter_mode=self.mode)
        )

        return self  # type: ignore

    def predict(
            self,
            X: pd.DataFrame,
        ) -> pd.Series:
        if (self.best_estimator_ is None) or (not hasattr(self.best_estimator_, "predict")):
            raise RuntimeError("Model is not refitted")
        return self.best_estimator_.predict(X)

    def predict_proba(
            self,
            X: pd.DataFrame
        ) -> pd.Series:
        if (self.best_estimator_ is None) or (not hasattr(self.best_estimator_, "predict_proba")):
            raise AttributeError("Underlying estimator does not support predict_proba().")
        return self.best_estimator_.predict_proba(X)

    def score(
            self,
            X: pd.DataFrame,
            y: pd.Series,
        ) -> float:

        if hasattr(self.best_estimator_, "score"):
            return self.best_estimator_.score(X, y)
        else:
            y_pred = self.predict(X)
            if is_classifier(self.best_estimator_):
                return accuracy_score(y, y_pred)
            elif is_regressor(self.best_estimator_):
                return r2_score(y, y_pred)
            else:
                raise TypeError(f"Unsupported estimator type: {type(self.best_estimator_).__name__}") 