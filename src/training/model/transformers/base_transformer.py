from typing import Any

from sklearn.base import TransformerMixin  # type: ignore # noqa
from sklearn.preprocessing import (  # type: ignore # noqa
    MinMaxScaler,
    QuantileTransformer,
    OneHotEncoder,
    FunctionTransformer,
    RobustScaler,
    StandardScaler,
    OrdinalEncoder,
)
from sklearn.impute import (  # type: ignore # noqa
    SimpleImputer,
)
from sklearn.pipeline import Pipeline  # type: ignore # noqa

from databricks_mlops_stack.utils.mlops_utils import ParserUtils  # type: ignore # noqa
from databricks_mlops_stack.training.model.transformers.log_transformer import LogTransformer  # type: ignore # noqa
from databricks_mlops_stack.training.model.transformers.clipper_transformer import ClipperTransformer  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.model import (  # type: ignore # noqa
    CONFIG_TRANSFORMER_CLASS,
    TRANSFORMER_MINMAX,
    TRANSFORMER_QUANTILE,
    TRANSFORMER_ONEHOT,
    TRANSFORMER_FUNCTION,
    TRANSFORMER_SIMPLEIMP,
    TRANSFORMER_LOG,
    TRANSFORMER_CLIP,
    TRANSFORMER_ROBUST,
    TRANSFORMER_STANDARD,
    TRANSFORMER_ORDINAL,
)

# -----------------------------------------------------------------------------
# Catalogs
# -----------------------------------------------------------------------------
AVAILABLE_TRANSFORMERS = {
    TRANSFORMER_MINMAX: MinMaxScaler,
    TRANSFORMER_QUANTILE: QuantileTransformer,
    TRANSFORMER_ONEHOT: OneHotEncoder,
    TRANSFORMER_FUNCTION: FunctionTransformer,
    TRANSFORMER_SIMPLEIMP: SimpleImputer,
    TRANSFORMER_LOG: LogTransformer,
    TRANSFORMER_CLIP: ClipperTransformer,
    TRANSFORMER_ROBUST: RobustScaler,
    TRANSFORMER_STANDARD: StandardScaler,
    TRANSFORMER_ORDINAL: OrdinalEncoder,
}

def get_one_transformer(
        transformer_class_name: str,
        transformer_params: dict | None,
    ) -> TransformerMixin:

    transformer_object = AVAILABLE_TRANSFORMERS.get(transformer_class_name.lower().strip(), None)
    if not transformer_object:
        raise ValueError(f"Unsupported transformer: {transformer_class_name}")
    
    if transformer_params:
        if not all([isinstance(this_key, str) for this_key in transformer_params]):
            raise ValueError(f"Transformer 'params' keys must be strings: {transformer_params}")
    else:
        transformer_params = dict()

    transformer_params = ParserUtils.parse_dictionary_params(transformer_params)
    final_transformer = transformer_object(**transformer_params)

    return final_transformer

def get_transformer(
        transformer_step_name: str,
        transformer_config: dict[str, Any],
    ) -> TransformerMixin:

    transformer_class_name = transformer_config.get(CONFIG_TRANSFORMER_CLASS, None)
    if not transformer_class_name:
        raise ValueError(f"Transformer class/name should be provide for: {transformer_step_name}")

    transformer_params = transformer_config.get("params")

    match (transformer_class_name, transformer_params):

        case [str() as unique_transformer, (dict() | None) as unique_params]:

            final_transformer = get_one_transformer(unique_transformer, unique_params)

        case [list() as list_transformer, (list() | None) as list_params] if (
            list_params is None or len(list_transformer) == len(list_params)
        ):

            list_params = list_params or [None] * len(list_transformer)

            list_one_transformers = []
            for unique_transformer, unique_params in zip(list_transformer, list_params):
                list_one_transformers.append(get_one_transformer(unique_transformer, unique_params))

            final_steps = [(f'{transformer_step_name}_{i}', t) for i, t in enumerate(list_one_transformers) ]
            final_transformer = Pipeline(final_steps)

        case [list(), _]:
            raise ValueError("When transformer 'type' is list, 'params' must be list of same length or None")

        case _:
            raise TypeError("Transformer 'type' must be str or list[str], 'params' must match structure")

    return final_transformer