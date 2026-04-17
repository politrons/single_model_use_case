import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from shared_utilities.utils import (
    get_config_file,
)


def _rename_if_exists(df: DataFrame, source: str, target: str) -> DataFrame:
    if source in df.columns and target not in df.columns:
        return df.withColumnRenamed(source, target)
    return df


def _ensure_col(df: DataFrame, column_name: str, spark_type: str):
    if column_name not in df.columns:
        return df.withColumn(column_name, F.lit(None).cast(spark_type))
    return df


def _normalize_target_col(target_col: F.Column) -> F.Column:
    target_clean = F.lower(F.regexp_replace(F.trim(target_col), r"[^a-z0-9]+", "_"))
    return (
        F.when(target_clean.isin("burning_cost", "bc"), F.lit("BC"))
        .when(target_clean.isin("frequency_claim", "frequency_claims"), F.lit("Frequency_claims"))
        .when(target_clean.isin("severity_claim", "severity_claims"), F.lit("Severity_claims"))
        .otherwise(F.col("target"))
    )


def _prepare_gold_features(df_gold: DataFrame) -> DataFrame:
    for source in ["year_month_date", "YearMonthDate"]:
        df_gold = _rename_if_exists(df_gold, source, "YearMonthDate")
    for source in ["nb_member", "NumberMembers"]:
        df_gold = _rename_if_exists(df_gold, source, "NumberMembers")
    for source in ["nb_claim", "NumberClaims"]:
        df_gold = _rename_if_exists(df_gold, source, "NumberClaims")
    for source in ["amount_paid", "AmountPaid"]:
        df_gold = _rename_if_exists(df_gold, source, "AmountPaid")
    for source in ["nb_invoice", "NumberInvoices"]:
        df_gold = _rename_if_exists(df_gold, source, "NumberInvoices")
    for source in ["frequency_claim", "Frequency_claims"]:
        df_gold = _rename_if_exists(df_gold, source, "Frequency_claims")
    for source in ["severity_claim", "Severity_claims"]:
        df_gold = _rename_if_exists(df_gold, source, "Severity_claims")
    for source in ["burning_cost", "BC"]:
        df_gold = _rename_if_exists(df_gold, source, "BC")
    for source in ["frequency_invoice", "Frequency_invoices"]:
        df_gold = _rename_if_exists(df_gold, source, "Frequency_invoices")
    for source in ["severity_invoice", "Severity_invoices"]:
        df_gold = _rename_if_exists(df_gold, source, "Severity_invoices")
    for source in ["CPI", "cpi"]:
        df_gold = _rename_if_exists(df_gold, source, "CPI")
    for source in ["price_CPI", "price_cpi"]:
        df_gold = _rename_if_exists(df_gold, source, "price_CPI")
    for source in ["SeenMonth", "seen_month"]:
        df_gold = _rename_if_exists(df_gold, source, "SeenMonth")

    for required_col, required_type in [
        ("YearMonthDate", "date"),
        ("cluster_id", "string"),
        ("NumberMembers", "double"),
        ("NumberClaims", "double"),
        ("AmountPaid", "double"),
        ("NumberInvoices", "double"),
        ("Frequency_claims", "double"),
        ("Severity_claims", "double"),
        ("BC", "double"),
        ("CPI", "double"),
        ("price_CPI", "double"),
        ("Frequency_invoices", "double"),
        ("Severity_invoices", "double"),
        ("SeenMonth", "string"),
    ]:
        df_gold = _ensure_col(df_gold, required_col, required_type)

    return df_gold.select(
        "cluster_id",
        "YearMonthDate",
        "NumberMembers",
        "NumberClaims",
        "AmountPaid",
        "NumberInvoices",
        "Frequency_claims",
        "Severity_claims",
        "BC",
        "CPI",
        "price_CPI",
        "Frequency_invoices",
        "Severity_invoices",
        "SeenMonth",
    )


def _prepare_cluster_dimensions() -> DataFrame:
    clusters_table = get_config_file("inflation", "run_config")["clusters_table"]
    df_cluster = spark.table(clusters_table)

    for source in ["ClusterId", "cluster_id", "segment"]:
        df_cluster = _rename_if_exists(df_cluster, source, "cluster_id")
    for source in ["MajorICDGroupingDescription", "major_icd_grouping_description"]:
        df_cluster = _rename_if_exists(df_cluster, source, "MajorICDGroupingDescription")
    for source in ["AgeBucket", "age_bucket"]:
        df_cluster = _rename_if_exists(df_cluster, source, "AgeBucket")
    for source in ["UKMemberRegion", "uk_member_region"]:
        df_cluster = _rename_if_exists(df_cluster, source, "UKMemberRegion")
    for source in ["PolicySubType", "policy_sub_type"]:
        df_cluster = _rename_if_exists(df_cluster, source, "PolicySubType")

    for required_col in [
        "cluster_id",
        "MajorICDGroupingDescription",
        "AgeBucket",
        "UKMemberRegion",
        "PolicySubType",
    ]:
        df_cluster = _ensure_col(df_cluster, required_col, "string")

    return df_cluster.select(
        "cluster_id",
        "MajorICDGroupingDescription",
        "AgeBucket",
        "UKMemberRegion",
        "PolicySubType",
    ).dropDuplicates(["cluster_id"])


@dp.materialized_view(name="inflation_predictions_export")
def inflation_predictions_export():
    df_raw = spark.table("inflation_predictions_raw")

    df_raw = df_raw.withColumn("target_normalized", _normalize_target_col(F.col("target")))

    grouping = ["cluster_id", "year_month_date", "origin"]
    df_pred = df_raw.groupBy(*grouping).agg(
        F.first(F.when(F.col("target_normalized") == "BC", F.col("prediction")), ignorenulls=True).alias("BC_pred"),
        F.first(F.when(F.col("target_normalized") == "BC", F.col("prediction_lower")), ignorenulls=True).alias("BC_pred_lower"),
        F.first(F.when(F.col("target_normalized") == "BC", F.col("prediction_upper")), ignorenulls=True).alias("BC_pred_upper"),
        F.first(F.when(F.col("target_normalized") == "Frequency_claims", F.col("prediction")), ignorenulls=True).alias("Frequency_claims_pred"),
        F.first(F.when(F.col("target_normalized") == "Frequency_claims", F.col("prediction_lower")), ignorenulls=True).alias("Frequency_claims_pred_lower"),
        F.first(F.when(F.col("target_normalized") == "Frequency_claims", F.col("prediction_upper")), ignorenulls=True).alias("Frequency_claims_pred_upper"),
        F.first(F.when(F.col("target_normalized") == "Severity_claims", F.col("prediction")), ignorenulls=True).alias("Severity_claims_pred"),
        F.first(F.when(F.col("target_normalized") == "Severity_claims", F.col("prediction_lower")), ignorenulls=True).alias("Severity_claims_pred_lower"),
        F.first(F.when(F.col("target_normalized") == "Severity_claims", F.col("prediction_upper")), ignorenulls=True).alias("Severity_claims_pred_upper"),
        F.first(F.when(F.col("target_normalized") == "BC", F.col("model_name")), ignorenulls=True).alias("_model_name_bc"),
        F.first("model_name", ignorenulls=True).alias("_model_name_any"),
        F.first(F.when(F.col("target_normalized") == "BC", F.col("model_config")), ignorenulls=True).alias("_model_config_bc"),
        F.first("model_config", ignorenulls=True).alias("_model_config_any"),
    ).withColumn("model_name", F.coalesce(F.col("_model_name_bc"), F.col("_model_name_any"))).withColumn(
        "model_config", F.coalesce(F.col("_model_config_bc"), F.col("_model_config_any"))
    ).drop("_model_name_bc", "_model_name_any", "_model_config_bc", "_model_config_any")

    df_gold = _prepare_gold_features(spark.table("inflation_gold"))
    df_cluster = _prepare_cluster_dimensions()

    df = (
        df_pred
        .join(
            df_gold,
            on=[
                df_pred["cluster_id"] == df_gold["cluster_id"],
                F.to_date(df_pred["year_month_date"]) == F.to_date(df_gold["YearMonthDate"]),
            ],
            how="left",
        )
        .drop(df_gold["cluster_id"])
        .join(df_cluster, on="cluster_id", how="left")
        .withColumn("segment", F.col("cluster_id"))
        .withColumn("YearMonthDate", F.coalesce(F.to_date("year_month_date"), F.to_date("YearMonthDate")))
    )

    for required_col, required_type in [
        ("BC_pred", "double"),
        ("BC_pred_lower", "double"),
        ("BC_pred_upper", "double"),
        ("Frequency_claims_pred", "double"),
        ("Frequency_claims_pred_lower", "double"),
        ("Frequency_claims_pred_upper", "double"),
        ("Severity_claims_pred", "double"),
        ("Severity_claims_pred_lower", "double"),
        ("Severity_claims_pred_upper", "double"),
        ("origin", "string"),
        ("model_name", "string"),
        ("model_config", "string"),
        ("segment", "string"),
        ("MajorICDGroupingDescription", "string"),
        ("AgeBucket", "string"),
        ("UKMemberRegion", "string"),
        ("PolicySubType", "string"),
    ]:
        df = _ensure_col(df, required_col, required_type)

    return df.select(
        "YearMonthDate",
        "NumberMembers",
        "NumberClaims",
        "AmountPaid",
        "NumberInvoices",
        "Frequency_claims",
        "Severity_claims",
        "BC",
        "CPI",
        "price_CPI",
        "BC_pred",
        "BC_pred_lower",
        "BC_pred_upper",
        "Frequency_claims_pred",
        "Frequency_claims_pred_lower",
        "Frequency_claims_pred_upper",
        "Severity_claims_pred",
        "Severity_claims_pred_lower",
        "Severity_claims_pred_upper",
        "segment",
        "model_name",
        "Frequency_invoices",
        "Severity_invoices",
        "SeenMonth",
        "origin",
        "model_config",
        "MajorICDGroupingDescription",
        "AgeBucket",
        "UKMemberRegion",
        "PolicySubType",
    )
