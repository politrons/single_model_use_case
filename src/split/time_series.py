import logging
from typing import Any

import pandas as pd   # type: ignore # noqa

from databricks_mlops_stack.split.split import Split   # type: ignore # noqa
from databricks_mlops_stack.utils.constants.core import (  # type: ignore # noqa
    X_TRAIN,
    X_VAL,
    X_TEST,
    Y_TRAIN,
    Y_VAL,
    Y_TEST,
    CONFIG_TEMPORAL_COLUMN_NAME,
    CONFIG_TRAIN_SIZE,
    CONFIG_TEST_SIZE,
    CONFIG_VAL_SIZE,
    CONFIG_CREATE_VALIDATION_SET,
    CONFIG_STRATIFY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOG = logging.getLogger("framework.split.time_series")


# Available split strategies
class TimeSeriesSplit(Split):

    def split(self, X: pd.DataFrame, y: pd.Series, split_config: dict[str, Any]) -> dict[str, Any]:
        """
           Split the data into train/test (and optionally validation) sets based on config.
           Ensures that in time_series strategy, date boundaries do not overlap between splits.
           """
        LOG.info("Split time series")
        # Extract config with defaults
        maybe_train_size: float | None = split_config.get(CONFIG_TRAIN_SIZE)
        maybe_test_size: float | None = split_config.get(CONFIG_TEST_SIZE)
        date_column: str | None = split_config.get(CONFIG_TEMPORAL_COLUMN_NAME)
        create_validation_set: bool = split_config.get(CONFIG_CREATE_VALIDATION_SET, False)
        val_size: float = split_config.get(CONFIG_VAL_SIZE, 0.1)
        stratify: bool | None = split_config.get(CONFIG_STRATIFY)

        train_size=0.0
        if maybe_train_size is not None and maybe_test_size is not None:
            raise ValueError("Provide either train_size OR test_size, not both.")
        if maybe_train_size is None and maybe_test_size is None:
            train_size = 0.9
        elif maybe_test_size is not None:
            train_size = 1 - maybe_test_size
        if not date_column:
            raise ValueError(f"{CONFIG_TEMPORAL_COLUMN_NAME} must be provided for time_series split.")
        if date_column not in X.columns:
            raise ValueError(f"{CONFIG_TEMPORAL_COLUMN_NAME} '{date_column}' not found in features.")
        
        if stratify:
            raise ValueError(f"{CONFIG_STRATIFY} cannot be set for time series.")

        X_sorted = X.sort_values(by=date_column)
        y_sorted = y.loc[X_sorted.index]

        unique_dates = X_sorted[date_column].sort_values().unique()

        cutoff_index = int(len(unique_dates) * train_size)
        train_dates = unique_dates[:cutoff_index]
        test_dates = unique_dates[cutoff_index:]

        if create_validation_set:
            cutoff_val_index = int(len(train_dates) * (1 - val_size))
            train_final_dates = train_dates[:cutoff_val_index]
            val_dates = train_dates[cutoff_val_index:]

            return {
                X_TRAIN: X_sorted[X_sorted[date_column].isin(train_final_dates)],
                X_VAL: X_sorted[X_sorted[date_column].isin(val_dates)],
                X_TEST: X_sorted[X_sorted[date_column].isin(test_dates)],
                Y_TRAIN: y_sorted.loc[X_sorted[date_column].isin(train_final_dates)],
                Y_VAL: y_sorted.loc[X_sorted[date_column].isin(val_dates)],
                Y_TEST: y_sorted.loc[X_sorted[date_column].isin(test_dates)],
            }
        else:
            return {
                X_TRAIN: X_sorted[X_sorted[date_column].isin(train_dates)],
                X_TEST: X_sorted[X_sorted[date_column].isin(test_dates)],
                Y_TRAIN: y_sorted.loc[X_sorted[date_column].isin(train_dates)],
                Y_TEST: y_sorted.loc[X_sorted[date_column].isin(test_dates)],
            }

build = TimeSeriesSplit()
