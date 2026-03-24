import logging

from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore # noqa
from imblearn.base import BaseSampler  # type: ignore # noqa
from sklearn.base import BaseEstimator  # type: ignore # noqa
from sklearn.compose import ColumnTransformer  # type: ignore # noqa
from sklearn.pipeline import Pipeline  # type: ignore # noqa

from databricks_mlops_stack.training.model.pipeline.early_stop_pipe import UniversalEarlyStoppingPipeline  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.model import (  # type: ignore # noqa
    STEP_MODEL,
    STEP_PREPROCESS,
    STEP_SAMPLER,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOG = logging.getLogger("framework.training.model.pipeline.base_pipe")

def build_pipeline_model(
        col_transform: ColumnTransformer,
        maybe_sampler: BaseSampler | None,
        has_early_stop_config: bool,
        model: BaseEstimator,
    ) -> Pipeline:
    """Construct the model pipeline (optionally with sampling).

    Parameters
    ----------
    col_transform : ColumnTransformer
        Preprocessing transformer.
    maybe_sampler : BaseSampler or None
        Sampler class if configured.
    has_early_stop_config : bool
        Flag to indicate if early stop is used.
    model : BaseEstimator
        The configured estimator.

    Returns
    -------
        A scikit-learn/imbalanced-learn/custom_early pipeline with preprocessing → [sampler] → model.
    """

    # define pipe class
    if has_early_stop_config:
        pipeline_class = UniversalEarlyStoppingPipeline
    elif maybe_sampler is not None:
        pipeline_class = ImbPipeline
    else:
        pipeline_class = Pipeline
    LOG.info(f'Pipe class: {pipeline_class}')

    # create steps
    steps = [(STEP_PREPROCESS, col_transform)]
    if maybe_sampler is not None:
        steps.append((STEP_SAMPLER, maybe_sampler))
    steps.append((STEP_MODEL, model))
    LOG.info(f'Steps to append: {steps}')

    # make final pipe
    final_pipe = pipeline_class(steps)

    return final_pipe