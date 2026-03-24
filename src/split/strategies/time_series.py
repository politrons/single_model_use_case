import logging
from typing import Any

import pandas as pd   # type: ignore # noqa

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
LOG = logging.getLogger("framework.split.time_series")

def split_by_timeseries(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float,
        val_size: float,
        create_validation_set: bool,
        train_cutoff_date: str,
        val_cutoff_date: str,
        date_column: str,
    ) -> dict[str, Any]:

    LOG.info("Starting time series split")

    # Sort data by date column and preserve alignment between X and y
    sort_indices = X.sort_values(by=date_column).index
    X_sorted = X.loc[sort_indices].reset_index(drop=True)
    y_sorted = y.loc[sort_indices].reset_index(drop=True)

    # Get unique dates (sorted)
    unique_dates = X_sorted[date_column].unique()
    
    LOG.info(f"Total samples: {len(X_sorted)}")
    LOG.info(f"Total unique dates: {len(unique_dates)}")

    # Step 1: Split into train_dates and test_dates
    if train_cutoff_date:
        LOG.info(f"Splitting at train_cutoff_date: {train_cutoff_date}")
        train_dates = unique_dates[unique_dates < train_cutoff_date]
        test_dates = unique_dates[unique_dates >= train_cutoff_date]
    else:
        train_size = 1.0 - test_size
        LOG.info(f"Using train_size: {train_size:.2%}")
        cutoff_idx = int(len(unique_dates) * train_size)
        train_dates = unique_dates[:cutoff_idx]
        test_dates = unique_dates[cutoff_idx:]

    LOG.info(f"Train dates: {len(train_dates)} dates from {train_dates.min()} to {train_dates.max()}")
    LOG.info(f"Test dates: {len(test_dates)} dates from {test_dates.min()} to {test_dates.max()}")

    # Step 2: Handle validation set if needed
    if create_validation_set:
        if val_cutoff_date:
            LOG.info(f"Splitting train at val_cutoff_date: {val_cutoff_date}")
            train_final_dates = train_dates[train_dates < val_cutoff_date]
            val_dates = train_dates[train_dates >= val_cutoff_date]
        else:
            LOG.info(f"Using val_size: {val_size:.2%}")
            cutoff_val_idx = int(len(train_dates) * (1 - val_size))
            train_final_dates = train_dates[:cutoff_val_idx]
            val_dates = train_dates[cutoff_val_idx:]

        LOG.info(f"Final train dates: {len(train_final_dates)} dates from {train_final_dates.min()} to {train_final_dates.max()}")
        LOG.info(f"Validation dates: {len(val_dates)} dates from {val_dates.min()} to {val_dates.max()}")

        # Create masks
        train_mask = X_sorted[date_column].isin(train_final_dates)
        val_mask = X_sorted[date_column].isin(val_dates)
        test_mask = X_sorted[date_column].isin(test_dates)

        # Extract datasets
        X_train = X_sorted[train_mask]
        X_val = X_sorted[val_mask]
        X_test = X_sorted[test_mask]

        y_train = y_sorted[train_mask]
        y_val = y_sorted[val_mask]
        y_test = y_sorted[test_mask]

        return {
            X_TRAIN: X_train,
            X_VAL: X_val,
            X_TEST: X_test,
            Y_TRAIN: y_train,
            Y_VAL: y_val,
            Y_TEST: y_test,
        }
    else:
        # No validation set
        train_mask = X_sorted[date_column].isin(train_dates)
        test_mask = X_sorted[date_column].isin(test_dates)

        X_train = X_sorted[train_mask]
        X_test = X_sorted[test_mask]

        y_train = y_sorted[train_mask]
        y_test = y_sorted[test_mask]

        return {
            X_TRAIN: X_train,
            X_TEST: X_test,
            Y_TRAIN: y_train,
            Y_TEST: y_test,
        }