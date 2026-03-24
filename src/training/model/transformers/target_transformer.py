from typing import Any
import logging

from sklearn.base import BaseEstimator  # type: ignore # noqa
from sklearn.compose import TransformedTargetRegressor  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.model import CONFIG_SECTION_TARGET_TRANSFORMER  # type: ignore # noqa
from databricks_mlops_stack.training.model.transformers.base_transformer import get_transformer  # type: ignore # noqa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOG = logging.getLogger("framework.training.model.transformers.target_transformer")

def build_target_transformer(
        config: dict[str, Any],
        pipe: BaseEstimator,
    ) -> BaseEstimator:

    assigned_target_transformer_config = config.get(CONFIG_SECTION_TARGET_TRANSFORMER)

    if assigned_target_transformer_config:
        LOG.info('Target transformer detected')

        target_tranformer = get_transformer(
            "get_target_transformer_dummy_name_not_used",
            assigned_target_transformer_config,
        )
        LOG.info(f'selected: {target_tranformer}')

        pipe = TransformedTargetRegressor(
            regressor=pipe,
            transformer=target_tranformer,
        )

    return pipe