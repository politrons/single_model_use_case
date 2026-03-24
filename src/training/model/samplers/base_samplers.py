from typing import Any
import logging

from imblearn.over_sampling import SMOTE  # type: ignore # noqa
from imblearn.base import BaseSampler  # type: ignore # noqa

from databricks_mlops_stack.utils.mlops_utils import ParserUtils  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.model import (  # type: ignore # noqa
    CONFIG_SECTION_SAMPLING,
    SAMPLING_SMOTE,
    CONFIG_SAMPLING_CLASS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOG = logging.getLogger("framework.training.model.samplers.base_samplers")

# -----------------------------------------------------------------------------
# Catalogs
# -----------------------------------------------------------------------------
AVAILABLE_SAMPLERS = {
    SAMPLING_SMOTE: SMOTE
}

def build_sampler_preprocessing(
        config: dict[str, Any],
    ) -> BaseSampler | None:

    sampling_config = config.get(CONFIG_SECTION_SAMPLING)
    if not sampling_config:
        return None

    sampler_class_name = sampling_config[CONFIG_SAMPLING_CLASS].lower().strip()
    LOG.info(f'Sampler class name: {sampler_class_name}')

    if sampler_class_name not in AVAILABLE_SAMPLERS:
        raise ValueError(f"Unsupported sampler: {sampler_class_name}")
    
    sampler_class = AVAILABLE_SAMPLERS[sampler_class_name]

    sampler_params = sampling_config.get("params", {})
    sampler_params = ParserUtils.parse_dictionary_params(sampler_params)

    sampler = sampler_class(**sampler_params)

    return sampler