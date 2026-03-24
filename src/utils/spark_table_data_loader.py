from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable, List, Tuple, cast

import pandas as pd  # type: ignore # noqa
from pyspark.sql import DataFrame, SparkSession  # type: ignore # noqa

from databricks_mlops_stack.utils.constants.core import (  # type: ignore # noqa
    CONFIG_FEATURE_COLUMNS,
    CONFIG_AUXILIARY_COLUMNS,
    CONFIG_FULL_TABLE_NAME,
    CONFIG_CATALOG_NAME,
    CONFIG_DEFAULT_CATALOG_NAME,
    CONFIG_SCHEMA_NAME,
    CONFIG_TABLE_NAME,
    CONFIG_TEMPORAL_COLUMN_NAME,
    CONFIG_RETRIEVAL_DATE_UPPER_VALUE,
    CONFIG_RETRIEVAL_DATE_LOWER_VALUE,
    CONFIG_TARGET_COLUMN,
)


def get_active_or_create_spark() -> SparkSession:
    """Return active Spark session or create one."""
    sess = SparkSession.getActiveSession()
    if sess is not None:
        return sess
    return SparkSession.builder.getOrCreate()


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return [str(x).strip() for x in list(v)]


class SparkTableDataLoader:
    """Shared Spark loader for config-driven table datasets."""

    def __init__(self, spark_getter: Callable[[], SparkSession] = get_active_or_create_spark):
        self._spark_getter = spark_getter

    def load_xy(self, args: dict[str, Any]) -> Tuple[pd.DataFrame | DataFrame, pd.Series | DataFrame]:
        explicit_catalog: str | None = args.get(CONFIG_CATALOG_NAME)
        default_catalog: str | None = args.get(CONFIG_DEFAULT_CATALOG_NAME)
        catalog: str | None = explicit_catalog or default_catalog
        schema: str | None = args.get(CONFIG_SCHEMA_NAME)
        table_name: str | None = args.get(CONFIG_TABLE_NAME)
        full_table_name: str | None = args.get(CONFIG_FULL_TABLE_NAME)

        features: List[str] = _as_list(args.get(CONFIG_FEATURE_COLUMNS))
        auxiliary_columns: List[str] = _as_list(args.get(CONFIG_AUXILIARY_COLUMNS))
        target: str = cast(str, args[CONFIG_TARGET_COLUMN])

        return_dataframe_type: str = str(args.get("return_dataframe_type", "pandas")).strip().lower()
        retrieval_date_upper: str | date | datetime | None = args.get(CONFIG_RETRIEVAL_DATE_UPPER_VALUE)
        retrieval_date_lower: str | date | datetime | None = args.get(CONFIG_RETRIEVAL_DATE_LOWER_VALUE)
        date_column: str | None = args.get(CONFIG_TEMPORAL_COLUMN_NAME)

        # Allow default_catalog to coexist with full_table_name because framework args
        # may always inject it. Only explicit table parts are considered conflicting.
        if full_table_name and any([explicit_catalog, schema, table_name]):
            raise ValueError(
                f"Provide either (catalog, schema, table_name) OR {CONFIG_FULL_TABLE_NAME}, not both."
            )
        if not full_table_name and not all([catalog, schema, table_name]):
            raise ValueError(
                f"You must provide either {CONFIG_FULL_TABLE_NAME} OR (catalog, schema, table_name)."
            )

        table = full_table_name or f"{catalog}.{schema}.{table_name}"
        if return_dataframe_type not in {"pandas", "spark"}:
            raise ValueError("return_dataframe_type must be either 'pandas' or 'spark'.")

        spark = self._spark_getter()
        df: DataFrame = spark.table(table)
        df = self._apply_temporal_filters(df, date_column, retrieval_date_lower, retrieval_date_upper)

        if return_dataframe_type == "pandas":
            pdf = df.toPandas()
            return pdf[features + auxiliary_columns], pdf[target]

        return df.select(*features, *auxiliary_columns), df.select(target)

    @staticmethod
    def _apply_temporal_filters(
        df: DataFrame,
        date_column: str | None,
        retrieval_date_lower: str | date | datetime | None,
        retrieval_date_upper: str | date | datetime | None,
    ) -> DataFrame:
        if not (retrieval_date_lower or retrieval_date_upper):
            return df

        if not date_column:
            raise ValueError(
                f"{CONFIG_TEMPORAL_COLUMN_NAME} must be provided when "
                f"using {CONFIG_RETRIEVAL_DATE_UPPER_VALUE} or {CONFIG_RETRIEVAL_DATE_LOWER_VALUE}."
            )

        if retrieval_date_lower:
            lower_bound = (
                retrieval_date_lower.strftime("%Y-%m-%d")
                if isinstance(retrieval_date_lower, (date, datetime))
                else retrieval_date_lower
            )
            df = df.where(df[date_column] >= lower_bound)

        if retrieval_date_upper:
            upper_bound = (
                retrieval_date_upper.strftime("%Y-%m-%d")
                if isinstance(retrieval_date_upper, (date, datetime))
                else retrieval_date_upper
            )
            df = df.where(df[date_column] <= upper_bound)

        return df
