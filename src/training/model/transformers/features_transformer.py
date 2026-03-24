from typing import Any
import logging

from sklearn.compose import ColumnTransformer  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.core import CONFIG_FEATURE_COLUMNS  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.model import (  # type: ignore # noqa
    CONFIG_SECTION_FEATURES_TRANSFORMERS,
    CONFIG_TRANSFORMER_FEATURES,
    CONFIG_SECTION_DISCARDED_FEATURES,
)
from databricks_mlops_stack.training.model.transformers.base_transformer import get_transformer  # type: ignore # noqa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOG = logging.getLogger("framework.training.model.transformers.features_transformers")

def build_features_preprocessing(
        config: dict[str, Any],
        is_neuralprophet: bool,
    ) -> ColumnTransformer:
    """Build a ``ColumnTransformer`` according to feature-level transformers.

    Configuration format
    --------------------
    ``features_transformers`` is a list of blocks like::

        - name: "scale_numeric"
          type: "MinMaxScaler" | "QuantileTransformer" | "OneHotEncoder"
          params: { ... }
          features: ["age", "income", ...]

    Any feature not explicitly transformed and not listed in ``discarded_features``
    will be passed through unchanged.

    Raises
    ------
    ValueError
        If a transformer references a discarded feature, or a transformer ``type`` is unknown.
    """
    transformers: list[tuple[str, Any, list[str]]] = []
    transformed_features: list[str] = []

    discarded_features = set(config.get(CONFIG_SECTION_DISCARDED_FEATURES, []))
    LOG.info(f"Feature transformer discarded_features: {discarded_features}")
    features = list(config.get(CONFIG_FEATURE_COLUMNS, []))
    LOG.info(f"Feature transformer features: {features}")

    for tconf in config.get(CONFIG_SECTION_FEATURES_TRANSFORMERS, []):

        this_name = tconf["name"]
        this_feature_tranformer = get_transformer(this_name, tconf)
        this_features = list(tconf[CONFIG_TRANSFORMER_FEATURES])

        for feature in this_features:
            if feature in discarded_features:
                raise ValueError(
                    f"Feature '{feature}' is listed in discarded_features but also assigned to a transformer"
                )

        transformers.append((this_name, this_feature_tranformer, this_features) )
        transformed_features.extend(this_features)

    passthrough_features = [f for f in features if f not in transformed_features and f not in discarded_features]
    if passthrough_features:
        LOG.info(f"passthrough features: {passthrough_features}")
        transformers.append(("passthrough", "passthrough", passthrough_features))

    final_preprocessing = ColumnTransformer(transformers, remainder="drop")
    if is_neuralprophet:
        LOG.info("NeuralProphet model detected, setting output to pandas")
        final_preprocessing.set_output(transform='pandas')

    return final_preprocessing