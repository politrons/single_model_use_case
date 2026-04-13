import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../shared"))
sys.path.insert(0, _shared)

from typing import Any
from pyspark.sql import functions as F
from pyspark.sql import Window as W
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import MapType, StringType
import json
import re
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from collections import defaultdict
from shared_utilities.utils import (
    get_config_file,
    _parse_str_to_bool,
    _parse_str_to_date,
)

LOWEST_GRANULARITY_PROPERTIES = {
    "PolicySubType": "contract",
    "AgeBucket": "contract",
    "UKMemberRegion": "contract",
    "SystemID": "invoice",
    "MajorICDGroupingDescription": "invoice",
    # "CareMarkerDescription": "invoice",
}


def sanitize_dbx_name(name: str) -> str:

    sanitized = re.sub(r"[.\s/()[\]{}|]|[\x00-\x1F\x7F]", "_", name)
    sanitized = sanitized.replace("-", "_")

    # Force lowercase (Unity Catalog stores names in lowercase)
    return sanitized.lower()


def sanitize_table_name(name: str) -> str:
    if len(name) > 255:
        raise ValueError("Databricks/UC only allow table names up to 255 chars")

    return sanitize_dbx_name(name)


def get_is_to_compensate_ibnr_on_invoices_view() -> bool:

    config = get_config_file("inflation", "run_config")

    return _parse_str_to_bool(config["is_to_compensate_ibnr_on_invoices_view"])


def get_timeframes(relative_path: str = "../../../") -> dict[str, Any]:

    config = get_config_file("inflation", "run_config", relative_path)

    timeframe = config["timeframe_info"]

    use_custom_ranges = timeframe["use_custom_ranges"]

    if use_custom_ranges:
        nb_forecast_years = timeframe["nb_forecast_years"]
        training_begin_range = _parse_str_to_date(timeframe["training_begin"])
        training_end_range = _parse_str_to_date(timeframe["training_end"])
        inference_begin_range = (training_end_range + relativedelta(months=1)).replace(day=1)
        inference_end_range = (inference_begin_range + relativedelta(years=nb_forecast_years) - relativedelta(months=1)).replace(day=1)
    else:
        input_date = datetime.today()
        nb_forecast_years = 3

        inference_begin_range = (input_date).replace(day=1).date()
        inference_end_range = (input_date + relativedelta(years=nb_forecast_years) - relativedelta(months=1)).replace(day=1).date()
        training_begin_range = date(2016, 1, 1)
        training_end_range = (input_date - relativedelta(months=1)).replace(day=1).date()

    all_timeframes = {
        "inference_begin_range": inference_begin_range,
        "inference_end_range": inference_end_range,
        "training_begin_range": training_begin_range,
        "training_end_range": training_end_range,
    }

    return all_timeframes


def compensate_ibnr_on_invoices_view(
    df_agg: DataFrame,
    df_ibnr_coeff: DataFrame,
    ibnr_segmentation: list,
) -> DataFrame:

    ibnr_targets = [
        ("ultimate_amount_paid", "coef_amount", "amount_paid"),
        ("ultimate_nb_claim_paid", "coef_nb_claim", "nb_claim"),
        ("ultimate_nb_invoice_paid", "coef_nb_invoice", "nb_invoice"),
    ]

    join_keys = [*ibnr_segmentation, "year_month_date"]
    df_agg_inflated = df_agg

    for target_value, coef_col, _ in ibnr_targets:
        df_coeff_subset = df_ibnr_coeff.filter(F.col("target") == target_value).withColumnRenamed("treatment_date", "year_month_date").select(*join_keys, coef_col)
        df_agg_inflated = table_merge(
            df_agg_inflated,
            df_coeff_subset,
            on=join_keys,
            how="outer",
        )

    for _, coef_col, metric_col in ibnr_targets:
        df_agg_inflated = df_agg_inflated.withColumn(
            metric_col,
            F.when(
                F.col(coef_col).isNotNull(),
                F.col(metric_col) * F.col(coef_col),
            ).otherwise(F.col(metric_col)),
        )

    return df_agg_inflated


def _create_derived_features(df: DataFrame) -> DataFrame:
    """Create derived features like frequency, severity, etc."""
    return (
        df.withColumn("frequency_claim", (F.col("nb_claim") / F.col("nb_member")))
        .withColumn("severity_claim", (F.col("amount_paid") / F.col("nb_claim")))
        .withColumn("frequency_invoice", (F.col("nb_invoice") / F.col("nb_member")))
        .withColumn("severity_invoice", (F.col("amount_paid") / F.col("nb_invoice")))
        .withColumn("burning_cost", (F.col("amount_paid") / F.col("nb_member")))
    )


def _join_with_regressors(df_main: DataFrame, df_regressors: DataFrame, temporal_col: str) -> DataFrame:
    """Join main dataframe with regressors and validate"""
    nb_observations = df_main.count()
    df_joined = df_main.join(df_regressors, on=temporal_col, how="inner")

    if df_joined.count() < nb_observations:
        raise ValueError("Not enough data in external regressors to compute forecast")

    return df_joined.orderBy(temporal_col, ascending=True)


def aggregate_and_process_claims_members_regressors(
    df_agg: DataFrame,
    temporal_col: str,
    cluster_id: str = None,
    cluster_filter: DataFrame = None,
    df_regressors: DataFrame = None,
) -> DataFrame:
    """Aggregate claims, members data and join with regressors"""

    if cluster_filter:
        null_counts = cluster_filter.select([F.count(F.when(F.col(c).isNotNull(), c)).alias(c) for c in cluster_filter.columns]).collect()[0]
        cols_to_keep = [c for c in cluster_filter.columns if null_counts[c] > 0]
        cluster_filter = cluster_filter.select(cols_to_keep)

        df = cluster_filter.join(df_agg, on=[c for c in cluster_filter.columns if c in df_agg.columns])
    else:
        df = df_agg

    combination = []

    # Claims aggregation
    agg_df1 = df
    for k in [k for k, v in LOWEST_GRANULARITY_PROPERTIES.items() if v == "contract" and k in df.columns]:
        has_other_values = agg_df1.filter(F.col(k) != "_").limit(1).count() > 0
        if has_other_values:
            agg_df1 = agg_df1.filter(F.col(k) != "_")
    agg_df1 = agg_df1.groupBy(temporal_col, *combination).agg(
        F.sum("amount_paid").alias("amount_paid"),
        F.sum("nb_claim").alias("nb_claim"),
        F.sum("nb_invoice").alias("nb_invoice"),
    )

    # Members aggregation
    agg_df2 = df.groupBy([temporal_col] + [k for k in df.columns if k in LOWEST_GRANULARITY_PROPERTIES and LOWEST_GRANULARITY_PROPERTIES[k] == "contract"]).agg(F.first("nb_member").alias("nb_member"))
    member_combination = [k for k in combination if LOWEST_GRANULARITY_PROPERTIES[k] == "contract"]
    removed_member_combination = [k for k, v in LOWEST_GRANULARITY_PROPERTIES.items() if k not in combination and v == "contract" and k in df.columns]
    for k in removed_member_combination:
        has_value = agg_df2.filter(F.col(k) == "_").limit(1).count() > 0
        if has_value:
            agg_df2 = agg_df2.filter(F.col(k) == "_")
    agg_df2 = agg_df2.groupBy([temporal_col] + member_combination).agg(F.first("nb_member").alias("nb_member"))

    # Join claims and members
    df_claims_members_agg = agg_df2.join(agg_df1, on=[temporal_col] + member_combination, how="left")
    df_claims_members_agg = df_claims_members_agg.fillna({"amount_paid": 0, "nb_claim": 0, "nb_invoice": 0})

    # Feature engineering
    df_claims_members_agg = _create_derived_features(df_claims_members_agg)

    # Join with regressors if provided
    if df_regressors is not None:
        return _join_with_regressors(df_claims_members_agg, df_regressors, temporal_col)

    df_claims_members_agg = df_claims_members_agg.withColumn("cluster_id", F.lit(cluster_id))
    return df_claims_members_agg.orderBy(temporal_col, ascending=True)
