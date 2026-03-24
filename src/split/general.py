import logging
from typing import Any

import pandas as pd  # type: ignore # noqa
from sklearn.model_selection import train_test_split  # type: ignore # noqa

from databricks_mlops_stack.split.split import Split  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.core import (  # type: ignore # noqa
    X_TRAIN,
    X_VAL,
    X_TEST,
    Y_TRAIN,
    Y_VAL,
    Y_TEST,
    CONFIG_RANDOM_STATE,
    CONFIG_TRAIN_SIZE,
    CONFIG_TEST_SIZE,
    CONFIG_VAL_SIZE,
    CONFIG_CREATE_VALIDATION_SET,
    CONFIG_STRATIFY,
    CONFIG_SHUFFLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOG = logging.getLogger("framework.split.general")


# Available split strategies
class GeneralSplit(Split):

    def split(self, X: pd.DataFrame, y: pd.Series, split_config: dict[str, Any]) -> dict[str, Any]:
        """
           Split the data into train/test (and optionally validation) sets based on config.
           Ensures that in time_series strategy, date boundaries do not overlap between splits.
           """
        LOG.info("Split general")
        # Extract config with defaults
        train_size: float | None = split_config.get(CONFIG_TRAIN_SIZE)
        test_size: float | None = split_config.get(CONFIG_TEST_SIZE)
        random_state: int | None = split_config.get(CONFIG_RANDOM_STATE)
        create_validation_set: bool = split_config.get(CONFIG_CREATE_VALIDATION_SET, False)
        val_size: float | None = split_config.get(CONFIG_VAL_SIZE)
        stratify: bool = split_config.get(CONFIG_STRATIFY, False)
        shuffle: bool = split_config.get(CONFIG_SHUFFLE, True)

        if train_size is not None and test_size is not None:
            raise ValueError(f"Provide either {CONFIG_TRAIN_SIZE} OR {CONFIG_TEST_SIZE}, not both.")
        if train_size is None and test_size is None:
            test_size = 0.1
        elif train_size is not None:
            test_size = 1 - train_size

        if random_state is None:
            raise ValueError(f"{CONFIG_RANDOM_STATE} must be provided for general split.")
        
        if not isinstance(stratify, bool):
            raise ValueError(f"{CONFIG_STRATIFY} must be either True or False (if not provided, defaults to False).")
        
        if not create_validation_set and val_size:
            raise ValueError(f"{CONFIG_CREATE_VALIDATION_SET} must be explicity set to True if {CONFIG_VAL_SIZE} is provided.")

        val_size = val_size or 0.1

        # for future implementations, this stratify_array can assume different values
        stratify_array = None
        if stratify:
            stratify_array = y

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_array,
        )

        if not create_validation_set:
            return {
                X_TRAIN: X_train_val,
                X_TEST: X_test,
                Y_TRAIN: y_train_val,
                Y_TEST: y_test,
            }
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify_array,
            )
            return {
                X_TRAIN: X_train,
                X_VAL: X_val,
                X_TEST: X_test,
                Y_TRAIN: y_train,
                Y_VAL: y_val,
                Y_TEST: y_test,
            }


build = GeneralSplit()

