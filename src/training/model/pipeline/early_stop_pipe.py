from typing import Any
import logging

import pandas as pd  # type: ignore # noqa
from sklearn.pipeline import Pipeline  # type: ignore # noqa
from sklearn.base import clone  # type: ignore # noqa
from sklearn.utils.validation import check_is_fitted  # type: ignore # noqa

from databricks_mlops_stack.utils.constants.model import (  # type: ignore # noqa
    STEP_MODEL,
    CONFIG_EVAL_SET,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOG = logging.getLogger("framework.training.model.pipeline.early_stop_pipe")

class UniversalEarlyStoppingPipeline(Pipeline):
    """
    Correct, safe Pipeline that supports early stopping for XGBoost, LightGBM, Keras, etc.
    - Final step must be named 'model'
    - All 'model__*' fit parameters are routed only to the final estimator
    - All preprocessing steps are properly fitted and stored
    - Works on any scikit-learn version

    CURRENTLY scikit metadata routing not working with eval sets, so this is needed

    """

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series| None = None,
            **fit_params,
        ) -> Pipeline:
        LOG.info("Running fit in UniversalEarlyStoppingPipeline")
        model_params = {}
        other_params = {}
        for key, value in fit_params.items():
            if key.startswith(f"{STEP_MODEL}__"):
                model_params[key[7:]] = value   # strip 'model__'
            else:
                other_params[key] = value

        # Clone the pipeline steps to avoid modifying the original
        self.steps: list = [(name, clone(step)) for name, step in self.steps]

        Xt, yt = X, y
        fitted_transformers = []

        # Fit all preprocessing steps and keep them
        for name, estimator in self.steps[:-1]:
            if hasattr(estimator, "fit_resample"):           # SMOTE, SMOTENC, etc.
                Xt, yt = estimator.fit_resample(Xt, yt, **other_params)
            else:
                Xt = estimator.fit_transform(Xt, yt, **other_params)
            fitted_transformers.append((name, estimator))

        # Now transform validation sets if present
        eval_set_transformed = None

        if CONFIG_EVAL_SET in model_params:
            eval_pairs = model_params.pop(CONFIG_EVAL_SET)
            eval_set_transformed = []
            for X_val, y_val in eval_pairs:
                X_val_t = X_val
                for _, trans in fitted_transformers:
                    if not hasattr(trans, "fit_resample"):   # skip resamplers on val
                        X_val_t = trans.transform(X_val_t)
                eval_set_transformed.append((X_val_t, y_val))

        # Fit the final model
        final_name, final_estimator = self.steps[-1]
        if eval_set_transformed is not None:
            final_estimator.fit(Xt, yt, eval_set=eval_set_transformed, **model_params)
        else:
            final_estimator.fit(Xt, yt, **model_params)

        # Store everything back
        self.steps = fitted_transformers + [(final_name, final_estimator)]

        return self

    def transform(
            self,
            X: pd.DataFrame,
        ) -> pd.DataFrame:

        check_is_fitted(self)

        Xt = X

        for _, step in self.steps[:-1]:
            if not hasattr(step, "fit_resample"):
                Xt = step.transform(Xt)

        return Xt

    def predict(
            self,
            X: pd.DataFrame,
            **predict_params: Any,
        ) -> pd.Series:
        check_is_fitted(self)
        Xt = self.transform(X)
        return self.steps[-1][1].predict(Xt, **predict_params)

    def predict_proba(
            self,
            X: pd.DataFrame,
            **predict_params: Any,
        ) -> pd.Series:
        check_is_fitted(self)
        Xt = self.transform(X)
        return self.steps[-1][1].predict_proba(Xt, **predict_params)
    
    def score(
            self,
            X: pd.DataFrame,
            y: pd.Series = None, 
            **score_params: Any,
        ) -> pd.Series:
        check_is_fitted(self)
        Xt = self.transform(X)
        return self.steps[-1][1].score(Xt, y, **score_params)