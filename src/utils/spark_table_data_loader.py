from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any, Callable, Dict, List, Tuple, cast

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
            available_columns = list(pdf.columns)
            x_columns = features + auxiliary_columns
            selected = self._select_with_flexible_names(pdf, x_columns, available_columns, "feature/auxiliary")
            resolved_target = self._resolve_target_column(target, available_columns)
            return selected, pdf[resolved_target]

        return df.select(*features, *auxiliary_columns), df.select(target)

    @staticmethod
    def _normalize_column_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(name).lower())

    @classmethod
    def _build_normalized_lookup(cls, available_columns: List[str]) -> Dict[str, List[str]]:
        lookup: Dict[str, List[str]] = {}
        for column in available_columns:
            normalized = cls._normalize_column_name(column)
            lookup.setdefault(normalized, []).append(column)
        return lookup

    @classmethod
    def _resolve_requested_column(
        cls,
        requested: str,
        available_columns: List[str],
        normalized_lookup: Dict[str, List[str]],
    ) -> str:
        if requested in available_columns:
            return requested

        lower_to_original = {column.lower(): column for column in available_columns}
        lower_match = lower_to_original.get(requested.lower())
        if lower_match:
            return lower_match

        normalized = cls._normalize_column_name(requested)
        normalized_matches = normalized_lookup.get(normalized, [])
        if len(normalized_matches) == 1:
            return normalized_matches[0]
        if len(normalized_matches) > 1:
            raise ValueError(
                f"Ambiguous requested column '{requested}'. Multiple matches found: {normalized_matches}."
            )
        raise KeyError(
            f"Column '{requested}' not found in source data. "
            f"Available columns: {available_columns}"
        )

    @classmethod
    def _select_with_flexible_names(
        cls,
        pdf: pd.DataFrame,
        requested_columns: List[str],
        available_columns: List[str],
        kind: str,
    ) -> pd.DataFrame:
        normalized_lookup = cls._build_normalized_lookup(available_columns)
        resolved_pairs: List[tuple[str, str]] = []
        missing: List[str] = []
        for requested in requested_columns:
            try:
                actual = cls._resolve_requested_column(requested, available_columns, normalized_lookup)
                resolved_pairs.append((requested, actual))
            except KeyError:
                missing.append(requested)

        if missing:
            raise KeyError(
                f"Missing {kind} columns: {missing}. "
                f"Available columns: {available_columns}"
            )

        selected_actual = [actual for _, actual in resolved_pairs]
        selected = pdf[selected_actual].copy()
        rename_map = {actual: requested for requested, actual in resolved_pairs}
        selected.rename(columns=rename_map, inplace=True)
        return selected

    @classmethod
    def _resolve_target_column(cls, target: str, available_columns: List[str]) -> str:
        normalized_lookup = cls._build_normalized_lookup(available_columns)
        return cls._resolve_requested_column(target, available_columns, normalized_lookup)

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
