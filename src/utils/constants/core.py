from typing import Final

# Config params - global
CONFIG_ENV: Final[str] = "env"
CONFIG_RANDOM_STATE: Final[str] = "random_state"
CONFIG_TEMPORAL_COLUMN_NAME: Final[str] = "temporal_reference_column"

# Config params - training data
CONFIG_TARGET_COLUMN: Final[str] = "target_column"
CONFIG_FEATURE_COLUMNS: Final[str] = "feature_columns"
CONFIG_AUXILIARY_COLUMNS: Final[str] = "auxiliary_columns"
CONFIG_FULL_TABLE_NAME: Final[str] = "full_table_name"
CONFIG_CATALOG_NAME: Final[str] = "catalog"
CONFIG_SCHEMA_NAME: Final[str] = "schema"
CONFIG_TABLE_NAME: Final[str] = "table_name"
CONFIG_DEFAULT_CATALOG_NAME: Final[str] = "default_catalog"
CONFIG_RETRIEVAL_DATE_UPPER_VALUE: Final[str] = "retrieval_date_upper"
CONFIG_RETRIEVAL_DATE_LOWER_VALUE: Final[str] = "retrieval_date_lower"

# Config params - split
CONFIG_TRAIN_SIZE: Final[str] = "train_size"
CONFIG_TEST_SIZE: Final[str] = "test_size"
CONFIG_VAL_SIZE: Final[str] = "val_size"
CONFIG_TRAIN_CUTOFF_DATE: Final[str] = "train_cutoff_date"
CONFIG_CREATE_VALIDATION_SET: Final[str] = "create_validation_set"
CONFIG_VAL_CUTOFF_DATE: Final[str] = "val_cutoff_date"
CONFIG_STRATIFY: Final[str] = "stratify"
CONFIG_SHUFFLE: Final[str] = "shuffle"
CONFIG_REMOVE_NULLS: Final[str] = "remove_target_nulls"

# Training data
X_TRAIN: Final[str] = "X_train"
X_VAL: Final[str] = "X_val"
X_TEST: Final[str] = "X_test"
Y_TRAIN: Final[str] = "y_train"
Y_VAL: Final[str] = "y_val"
Y_TEST: Final[str] = "y_test"

__all__ = [
    "CONFIG_ENV",
    "CONFIG_RANDOM_STATE",
    "CONFIG_RANDOM_STATE",
    "CONFIG_TEMPORAL_COLUMN_NAME",
    "CONFIG_TARGET_COLUMN",
    "CONFIG_FEATURE_COLUMNS",
    "CONFIG_AUXILIARY_COLUMNS",
    "CONFIG_FULL_TABLE_NAME",
    "CONFIG_CATALOG_NAME",
    "CONFIG_SCHEMA_NAME",
    "CONFIG_TABLE_NAME",
    "CONFIG_DEFAULT_CATALOG_NAME",
    "CONFIG_RETRIEVAL_DATE_UPPER_VALUE",
    "CONFIG_RETRIEVAL_DATE_LOWER_VALUE",
    "CONFIG_TRAIN_SIZE",
    "CONFIG_TEST_SIZE",
    "CONFIG_VAL_SIZE",
    "CONFIG_TRAIN_CUTOFF_DATE",
    "CONFIG_CREATE_VALIDATION_SET",
    "CONFIG_VAL_CUTOFF_DATE",
    "CONFIG_STRATIFY",
    "CONFIG_SHUFFLE",
    "CONFIG_REMOVE_NULLS",
    "X_TRAIN",
    "X_VAL",
    "X_TEST",
    "Y_TRAIN",
    "Y_VAL",
    "Y_TEST",
]