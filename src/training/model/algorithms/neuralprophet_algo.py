from typing import Any, Self
import random

from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore # noqa
import pandas as pd  # type: ignore # noqa
import numpy as np  # type: ignore # noqa
from neuralprophet import NeuralProphet  # type: ignore # noqa
import torch   # type: ignore # noqa
import pytorch_lightning as pl  # type: ignore # noqa

from databricks_mlops_stack.utils.constants.model import (  # type: ignore # noqa
    CONFIG_EPOCHS,
    CONFIG_BATCH_SIZE,
    CONFIG_EVAL_SET,
    TF_PATIENCE,
    TF_MIN_DELTA,
    FREQUENCY,
    LAGGED_REGRESSORS,
    FUTURE_REGRESSORS,
    EVENTS,
    COUNTRY_HOLIDAYS,
    SEASONALITY,
    CONFIG_ID_COLUMN,
    TRANSFORMER_ONEHOT,
)
from databricks_mlops_stack.utils.constants.core import (  # type: ignore # noqa
    CONFIG_TEMPORAL_COLUMN_NAME,
    CONFIG_RANDOM_STATE,
    CONFIG_FEATURE_COLUMNS,
)

# -----------------------------------------------------------------------------
# Catalogs and constants
# -----------------------------------------------------------------------------
_TEMPORAL_COLUMN = 'ds'

_ID_COLUMN = 'ID'

_SINGLE_ID_VALUE = 'single_timeseries'

_TARGET_COLUMN = 'y'

_PREDICTION_COLUMN_SUFFIX = 'yhat'

_REGISTRATION_METHODS: dict[str, str] = {
    LAGGED_REGRESSORS: "add_lagged_regressor",
    FUTURE_REGRESSORS: "add_future_regressor",
    EVENTS: "add_events",
    COUNTRY_HOLIDAYS: "add_country_holidays",
    SEASONALITY: "add_seasonality",
}

class NeuralProphetRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for NeuralProphet regression.
    Follows exactly the same API as RandomForestRegressor / XGBRegressor.
    """
    def __init__(
            self,
            config: dict[str, Any],
            random_state: int,
            params: dict,
            extra_params: dict,
        ) -> None:

        self.config: dict[str, Any] = config
        print(f"NeuralProphet model config: {self.config}")
        self.params: dict = params
        print(f"NeuralProphet model params: {self.params}")
        self.extra_params: dict = extra_params
        print(f"NeuralProphet model extra params: {self.extra_params}")

        self._set_all_seeds(random_state)

        self._hijack_config()

        self.epochs_completed: int = -1

        self.frequency: str = self.config.get(FREQUENCY, 'auto')
        print(f"NeuralProphet model frequency: '{self.frequency}'")

        self.id_column: str = self.config.get(CONFIG_ID_COLUMN, _ID_COLUMN)
        print(f"NeuralProphet model ID column: '{self.id_column}'")

        temporal_column_from_upper: str | None = self.extra_params.get(CONFIG_TEMPORAL_COLUMN_NAME)
        temporal_column_from_config: str | None = self.config.get(CONFIG_TEMPORAL_COLUMN_NAME)
        if temporal_column_from_upper and temporal_column_from_config:
            if temporal_column_from_upper != temporal_column_from_config:
                raise ValueError(f"Different {CONFIG_TEMPORAL_COLUMN_NAME} set at training/split and NeuralProphet config")
        self.temporal_column: str = temporal_column_from_upper or temporal_column_from_config or _TEMPORAL_COLUMN
        print(f"NeuralProphet model temporal column: '{self.temporal_column}'")

        self.original_features: list[str] = self.extra_params.get(CONFIG_FEATURE_COLUMNS, [])
        print(f"NeuralProphet model original features: {self.original_features}")

        for transformer in self.extra_params.get('all_feature_transformers', []):
            if transformer in [TRANSFORMER_ONEHOT]:
                raise ValueError(f"Currently the transformer {transformer} is not supported for NeuralProphet models")

        self._build_model()

        return

    def _build_model(self) -> None:

        if CONFIG_RANDOM_STATE in self.params:
            self.params.pop(CONFIG_RANDOM_STATE)
        self.model = NeuralProphet(**self.params)

        self._resolve_registration_methods()

        return

    def fit(
            self,
            X: pd.DataFrame,
            y: np.ndarray,
        ) -> None:
        print("NeuralProphet model: starting fit ...")

        df = self._validate_df(X, y)

        self._get_count_by_id(df, 'fit', 3)
        
        df = self._sort_df(df)

        self.model.fit(
            df,
            freq=self.frequency,
            continue_training=False,
            metrics=False,
        )

        self._prepare_history(df)

        return

    def predict(
            self,
            X: pd.DataFrame,
        ) -> np.ndarray:
        print("NeuralProphet model: starting predict ...")

        self.model.trainer = pl.Trainer(
            accelerator='auto',
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )

        df: pd.DataFrame = self._validate_df(X, None)

        self._get_count_by_id(df, 'predict')
        rows_order_sent = df.loc[:, [_ID_COLUMN, _TEMPORAL_COLUMN]].copy()

        ###############################################################################################
        ###############################################################################################
        print('history')
        print(self._history_df)
        print('df')
        print(df)
        ###############################################################################################
        ###############################################################################################

        df = self._make_future_dataframe(df)

        ###############################################################################################
        ###############################################################################################
        print('padded')
        print(df)
        ###############################################################################################
        ###############################################################################################

        forecast: pd.DataFrame = self.model.predict(df)

        forecast = (
            rows_order_sent
            .merge(
                forecast,
                on=[_ID_COLUMN, _TEMPORAL_COLUMN],
                how='left',
            )
        )

        pred_cols: list[str] = sorted(
            [c for c in forecast.columns if c.startswith(_PREDICTION_COLUMN_SUFFIX)],
            key=lambda c: (
                int(c.replace(_PREDICTION_COLUMN_SUFFIX, ""))
                if c.replace(_PREDICTION_COLUMN_SUFFIX, "").isdigit()
                else 0
            ),
        )

        if len(pred_cols) == 1:
            preds = forecast[pred_cols[0]]
        elif len(pred_cols) > 1:
            preds = forecast[pred_cols].mean(axis=1)
        else:
            preds = [np.nan] * len(forecast)
        preds = np.asarray(preds)

        print(f"NeuralProphet model count prediction not NaN: {np.count_nonzero(~np.isnan(preds)):,}")
        ###############################################################################################
        print(f"NeuralProphet model warn: {np.count_nonzero(~np.isnan(preds)):,} != {len(preds)}")
        ###############################################################################################

        return preds

    def _set_all_seeds(
            self,
            seed: int,
        ) -> None:

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

        self.random_state: int = seed

        return

    def _hijack_config(self) -> None:
        """
        Since in TFBase some params can be passed at model level,
        to maintain consistency and design pattern,
        if passed here will be set directly on params.
        """

        configparams_to_hijack = [
            CONFIG_EPOCHS,
            CONFIG_BATCH_SIZE,
        ]

        for configparam in configparams_to_hijack:
            from_config: Any = self.config.get(configparam)
            from_params: Any = self.params.get(configparam)
            if from_config:
                print(f"NeuralProphet model detected {configparam} at config level: {from_config}")
                if from_params and from_config != from_params:
                    raise ValueError(f"2 different values detected at config/params for {configparam}")
                print(f"NeuralProphet model setting {configparam}: {from_config}")
                self.params.update({configparam: from_config})

        return

    def _resolve_registration_methods(self) -> None:

        for config_key, method_name in _REGISTRATION_METHODS.items():
            items: list[Any] | None = self.config.get(config_key)
            if items is None:
                print(f"NeuralProphet model no configuration set for {config_key}")
                continue
            if not isinstance(items, list):
                print(f"NeuralProphet model normalising {config_key} into list")
                items = [items]

            method = getattr(self.model, method_name)
            for item in items:
                match item:
                    case str() as item_str:
                        print(f"NeuralProphet model adding {config_key}: {item_str}")
                        method(item_str)
                    case dict() as item_dict:
                        print(f"NeuralProphet model adding {config_key}: {item_dict}")
                        method(**item_dict)
                    case _:
                        raise ValueError(f"Unsupported type in {config_key}: {type(item)}")

        return

    def _get_count_by_id(
            self,
            df: pd.DataFrame,
            called_function: str,
            check_count_by: int | None = None,
        ) -> None:

        count_by_id = df.groupby(_ID_COLUMN)[_TEMPORAL_COLUMN].count().reset_index()
        print(f"NeuralProphet model count by id in {called_function}: {count_by_id.set_index(_ID_COLUMN)[_TEMPORAL_COLUMN].to_dict()}")

        if check_count_by:
            ids_with_insufficient_rows = count_by_id.loc[count_by_id[_TEMPORAL_COLUMN] < check_count_by, _ID_COLUMN]
            if len(ids_with_insufficient_rows) > 0:
                raise ValueError(f"NeuralProphet model IDs with insufficent rows: '{set(ids_with_insufficient_rows)}'")

        return

    def _validate_df(
            self,
            X: pd.DataFrame,
            y: np.ndarray | None,
        ) -> pd.DataFrame:

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be a pandas DataFrame (required for NeuralProphet), not {type(X)}")

        df = X.copy()

        reconstructed_columns: list[str] = []
        for col in df.columns:
            for part in col.split('__'):
                if part in self.original_features:
                    reconstructed_columns.append(part)
        if set(reconstructed_columns) != set(self.original_features):
            print(f'NeuralProphet model reconstructed features: {reconstructed_columns}')
            print(f'NeuralProphet model original features: {self.original_features}')
            raise ValueError("NeuralProphet model something went wrong with column's name reconstruction")
        print(f'NeuralProphet model columns: {reconstructed_columns}')
        df.columns = reconstructed_columns

        if self.temporal_column not in df.columns:
            raise ValueError(f"NeuralProphet model temporal_column '{self.temporal_column}' not found in dataframe")
        df[self.temporal_column] = pd.to_datetime(df[self.temporal_column])

        if self.id_column not in df.columns:
            if self.id_column == _ID_COLUMN:
                print("NeuralProphet model ID column not found in training dataframe, adding single value")
                df[_ID_COLUMN] = _SINGLE_ID_VALUE
            else:
                raise ValueError(f"NeuralProphet model id column provided '{self.id_column}' not found in dataframe")

        df = df.rename(columns={
            self.temporal_column: _TEMPORAL_COLUMN,
            self.id_column: _ID_COLUMN,
        })
        df[_TARGET_COLUMN] = y

        return df
    
    def _sort_df(
            self,
            df: pd.DataFrame,
            first_by_id: bool = True,
        ) -> pd.DataFrame:

        if first_by_id:
            df = (
                df
                .sort_values([_ID_COLUMN, _TEMPORAL_COLUMN], ascending=[True, True])
                .reset_index(drop=True)
            )
        else:
            df = (
                df
                .sort_values([_TEMPORAL_COLUMN, _ID_COLUMN], ascending=[True, True])
                .reset_index(drop=True)
            )

        return df

    def _prepare_history(
            self,
            df: pd.DataFrame,
        ) -> None:

        self.unique_ids: list = list(df[_ID_COLUMN].unique())
        print(f"NeuralProphet unique IDs: {self.unique_ids}")
        self.number_unique_ids: int = len(self.unique_ids)
        print(f"NeuralProphet model number of unique IDs: {self.number_unique_ids}")

        df = self._sort_df(df, first_by_id=False)

        print(f"NeuralProphet model max lags: {self.model.max_lags}")
        history_length: int = self.number_unique_ids * self.model.max_lags
        if history_length == 0:
            history_length = 4 * self.number_unique_ids
        print(f"NeuralProphet model history length to be stored: {history_length}")

        self._history_df: pd.DataFrame = df.iloc[-history_length:].copy()
        self._history_df = self._sort_df(self._history_df)

        self._get_count_by_id(self._history_df, 'prepare_history', self.model.max_lags)

        return

    def _make_future_dataframe(
            self,
            df: pd.DataFrame,
        ) -> pd.DataFrame:

        make_future_dataframe_params = {
            'df': self._history_df,
            'periods': len(df),
            'n_historic_predictions': False,
        }

        future_regressors = self.model.config_regressors.regressors
        if future_regressors is not None and len(future_regressors.keys()) > 0:
            future_regressors = list(future_regressors.keys())
            print(f"NeuralProphet model adding future regresseors to make future dataframe: {future_regressors}")
            future_regressors.append(_TEMPORAL_COLUMN)
            make_future_dataframe_params['regressors_df'] = df[future_regressors].copy()

        padded = self.model.make_future_dataframe(**make_future_dataframe_params)
        print("NeuralProphet model history appended")

        padded = self._sort_df(padded)

        unique_ids = list(df[_ID_COLUMN].unique())
        new_ids = [iid for iid in unique_ids if iid not in self.unique_ids]
        if len(new_ids) > 0:
            raise ValueError(f"NeuralProphet model has ids not seen on train: {new_ids}")

        return padded

    def get_params(
            self,
            deep: bool = True,
        ) -> dict[str, Any]:
        print("NeuralProphet model getting params...")

        params = {"random_state": self.random_state}
        for key, value in self.params.items():
            params[f"params__{key}"] = value
        for key, value in self.config.items():
            params[f"config__{key}"] = value
        for key, value in self.extra_params.items():
            params[f"extra_params__{key}"] = value
        return params

    def set_params(
            self,
            **params: Any,
        ) -> Self:
        print("NeuralProphet model setting params...")

        params_updated = False
        config_updated = False
        extra_params_updated = False
        new_params = self.params.copy()
        new_config = self.config.copy()
        new_extra_params = self.extra_params.copy()

        for key, value in params.items():
            if key == "random_state":
                self.random_state = value
            elif key.startswith("params__"):
                subkey = key[8:]  # Strip 'params__'
                new_params[subkey] = value
                params_updated = True
            elif key.startswith("config__"):
                subkey = key[8:]  # Strip 'config__'
                new_config[subkey] = value
                config_updated = True
            elif key.startswith("extra_params__"):
                subkey = key[14:]  # Strip 'extra_params__'
                new_extra_params[subkey] = value
                extra_params_updated = True
            else:
                raise ValueError(f"Unknown parameter: {key}")

        if params_updated:
            self.params = new_params
        if config_updated:
            self.config = new_config
        if extra_params_updated:
            self.extra_params = new_extra_params
        if params_updated or config_updated or extra_params_updated:
            self._build_model()

        return self