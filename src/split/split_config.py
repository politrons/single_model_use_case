import logging
from typing import Any

import pandas as pd  # type: ignore # noqa

from databricks_mlops_stack.split.strategies.general import split_by_iid  # type: ignore # noqa
from databricks_mlops_stack.split.strategies.time_series import split_by_timeseries  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.core import (  # type: ignore # noqa
    CONFIG_RANDOM_STATE,
    CONFIG_TRAIN_SIZE,
    CONFIG_TEST_SIZE,
    CONFIG_VAL_SIZE,
    CONFIG_CREATE_VALIDATION_SET,
    CONFIG_STRATIFY,
    CONFIG_SHUFFLE,
    CONFIG_TRAIN_CUTOFF_DATE,
    CONFIG_VAL_CUTOFF_DATE,
    CONFIG_TEMPORAL_COLUMN_NAME,
    CONFIG_REMOVE_NULLS,
)
from databricks_mlops_stack.utils.constants.model import (  # type: ignore # noqa
    CONFIG_SPLIT_STRATEGY,
    SPLIT_GENERAL,
    SPLIT_TIME,
    SPLIT_STRATIFY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOG = logging.getLogger("framework.split.split_config")

# -----------------------------------------------------------------------------
# Catalogs
# -----------------------------------------------------------------------------
AVAILABLE_SPLIT_STRATEGIES = {
    SPLIT_GENERAL,
    SPLIT_TIME,
    SPLIT_STRATIFY,
}

class SplitContractConfig:

    def split(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            split_config: dict[str, Any],
        ) -> dict[str, Any]:
        """
        Split the data into train/test (and optionally validation) sets based on config.
        Ensures that in time_series strategy, date boundaries do not overlap between splits.
        """
        LOG.info("Split by configuration")

        ######################################
        # extract config with (possible) defaults
        ######################################
        strategy_name: str = split_config.get(CONFIG_SPLIT_STRATEGY, SPLIT_GENERAL).lower().strip()
        LOG.info(f"{CONFIG_SPLIT_STRATEGY} set to '{strategy_name}'")

        create_validation_set: bool = split_config.get(CONFIG_CREATE_VALIDATION_SET, False)
        LOG.info(f"{CONFIG_CREATE_VALIDATION_SET} set to '{create_validation_set}'")

        stratify: bool = split_config.get(CONFIG_STRATIFY, False)
        LOG.info(f"{CONFIG_STRATIFY} set to '{stratify}'")

        shuffle: bool = split_config.get(CONFIG_SHUFFLE, True)
        LOG.info(f"{CONFIG_SHUFFLE} set to '{shuffle}'")

        remove_rows_where_target_isnull: bool = split_config.get(CONFIG_REMOVE_NULLS, True)
        LOG.info(f"{CONFIG_REMOVE_NULLS} set to '{remove_rows_where_target_isnull}'")

        train_size: float | None = split_config.get(CONFIG_TRAIN_SIZE)

        test_size: float | None = split_config.get(CONFIG_TEST_SIZE)

        val_size: float | None = split_config.get(CONFIG_VAL_SIZE)

        random_state: int | None = split_config.get(CONFIG_RANDOM_STATE)

        train_cutoff_date: str | None = split_config.get(CONFIG_TRAIN_CUTOFF_DATE)

        val_cutoff_date: str | None = split_config.get(CONFIG_VAL_CUTOFF_DATE)

        date_column: str | None = split_config.get(CONFIG_TEMPORAL_COLUMN_NAME)

        ######################################
        # basic validation
        ######################################
        if strategy_name not in AVAILABLE_SPLIT_STRATEGIES:
            raise ValueError(f"Unsupported {CONFIG_SPLIT_STRATEGY}: '{strategy_name}'")

        if train_size and test_size:
            raise ValueError(f"Provide either {CONFIG_TRAIN_SIZE} OR {CONFIG_TEST_SIZE}, not both.")

        if not create_validation_set and (val_size or val_cutoff_date):
            error_msg = f"{CONFIG_CREATE_VALIDATION_SET} must be explicity set to True"
            error_msg += f" if {CONFIG_VAL_SIZE}/{CONFIG_VAL_CUTOFF_DATE} is provided."
            raise ValueError(error_msg)

        if strategy_name in [SPLIT_GENERAL, SPLIT_STRATIFY]:

            if not random_state or not isinstance(random_state, int):
                raise ValueError(f"{CONFIG_RANDOM_STATE} (int) must be provided for '{strategy_name}'.")

            if not isinstance(stratify, bool):
                raise ValueError(f"Currently {CONFIG_STRATIFY} is only supported as True/False.")
            
            if date_column:
                LOG.info(f"{CONFIG_TEMPORAL_COLUMN_NAME} set but ignored for '{strategy_name}'")

            if train_cutoff_date:
                LOG.info(f"{CONFIG_TRAIN_CUTOFF_DATE} set but ignored for '{strategy_name}'")

            if val_cutoff_date:
                LOG.info(f"{CONFIG_VAL_CUTOFF_DATE} set but ignored for '{strategy_name}'")

            if strategy_name == SPLIT_STRATIFY and not stratify:
                stratify = True
                LOG.info(f"Overwriting {CONFIG_STRATIFY} to True to maintain expected behaviour")

        if strategy_name in [SPLIT_TIME]:

            if stratify:
                raise ValueError(f"{CONFIG_STRATIFY} cannot be set for '{SPLIT_TIME}'.")

            if random_state:
                LOG.info(f"{CONFIG_RANDOM_STATE} set but ignored for '{SPLIT_TIME}'")

            if not date_column:
                raise ValueError(f"{CONFIG_TEMPORAL_COLUMN_NAME} must be provided for '{SPLIT_TIME}'.")

            if date_column not in X.columns:
                raise ValueError(f"{CONFIG_TEMPORAL_COLUMN_NAME} '{date_column}' not found in features.")
            
            if train_cutoff_date and (train_size or test_size):
                raise ValueError(f"Provide either {CONFIG_TRAIN_CUTOFF_DATE} OR {CONFIG_TRAIN_SIZE}/{CONFIG_TEST_SIZE}, not both.")
            
            if val_cutoff_date and val_size:
                raise ValueError(f"Provide either {CONFIG_VAL_CUTOFF_DATE} OR {CONFIG_VAL_SIZE}, not both.")

        ######################################
        # init and fallbacks
        ######################################
        if train_size is None and test_size is None:
            test_size = 0.1
        elif train_size:
            test_size = 1.0 - train_size
        LOG.info(f"{CONFIG_TEST_SIZE} set to {test_size:.2%}")

        val_size = val_size or 0.1
        LOG.info(f"{CONFIG_VAL_SIZE} set to {val_size:.2%}")

        ######################################
        # Remove target nulls
        ######################################
        if remove_rows_where_target_isnull:
            LOG.info('Requested to remove rows where target is NULL ...')
            rows_before = len(y)
            LOG.info(f'Rows before removing null targets: {rows_before:,}')

            mask = y.notna()

            X = X.loc[mask].reset_index(drop=True)
            y = y.loc[mask].reset_index(drop=True)

            rows_after = len(y)
            LOG.info(f'Rows after removing null targets: {rows_after:,}')
        else:
            LOG.info('Not requested to remove rows where target is NULL')

        ######################################
        # select requested strategy
        ######################################
        final_datasets = {}
        if strategy_name in [SPLIT_TIME]:
            final_datasets = split_by_timeseries(
                X=X,
                y=y,
                test_size=test_size,
                val_size=val_size,
                create_validation_set=create_validation_set,
                train_cutoff_date=train_cutoff_date,
                val_cutoff_date=val_cutoff_date,
                date_column=date_column,
            )
        else: # fallback
            final_datasets = split_by_iid(
                X=X,
                y=y,
                test_size=test_size,
                val_size=val_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify,
                create_validation_set=create_validation_set,
            )

        for k, v in final_datasets.items():
            v_shape = v.shape
            if len(v_shape) == 2:
                number_rows = v_shape[0]
                number_cols = v_shape[1]
                LOG.info(f"Final {k}\tshape: ({number_rows:,}, {number_cols:,})")

        return final_datasets

build = SplitContractConfig()
