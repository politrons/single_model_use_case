import logging
from typing import Any

import pandas as pd  # type: ignore # noqa
from sklearn.model_selection import train_test_split  # type: ignore # noqa

from databricks_mlops_stack.utils.constants.core import (  # type: ignore # noqa
    X_TRAIN,
    X_VAL,
    X_TEST,
    Y_TRAIN,
    Y_VAL,
    Y_TEST,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOG = logging.getLogger("framework.split.general")

def split_by_iid(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float,
        val_size: float,
        random_state: int,
        shuffle: bool,
        stratify: bool,
        create_validation_set: bool,
    ) -> dict[str, Any]:

    LOG.info("Starting split general (IID, a.k.a 'cross-sectional')")

    # for future implementations, this stratify_array can assume different values
    stratify_array = None
    if stratify:
        stratify_array = y
        LOG.info("Array to stratify by: target")

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