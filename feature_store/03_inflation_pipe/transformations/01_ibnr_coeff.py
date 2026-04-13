from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql import Window as W
from datetime import datetime


def _rename_if_exists(df, source: str, target: str):
    if source in df.columns and target not in df.columns:
        return df.withColumnRenamed(source, target)
    return df


def _normalize_ibnr_columns(df):
    # Compatibility with exports where fields are still CamelCase.
    df = _rename_if_exists(df, "Segment", "segment")
    df = _rename_if_exists(df, "FullExperiment", "full_experiment")
    df = _rename_if_exists(df, "Target", "target")
    df = _rename_if_exists(df, "TreatmentDate", "treatment_date")
    df = _rename_if_exists(df, "CumulativeAmountPaid", "cumulative_amount_paid")
    df = _rename_if_exists(df, "CumulativeNumberClaimsPaid", "cumulative_nb_claim_paid")
    df = _rename_if_exists(df, "CumulativeNumberInvoicesPaid", "cumulative_nb_invoice_paid")
    df = _rename_if_exists(df, "UltimateAmountPaidPredicted", "ultimate_amount_paid_predicted")
    df = _rename_if_exists(df, "UltimateNumberClaimsPaidPredicted", "ultimate_nb_claim_paid_predicted")
    df = _rename_if_exists(df, "UltimateNumberInvoicesPaidPredicted", "ultimate_nb_invoice_paid_predicted")
    # Alternative naming seen in some IBNR exports.
    df = _rename_if_exists(df, "UltimateChargePredicted", "ultimate_amount_paid_predicted")
    df = _rename_if_exists(df, "UltimateNumberClaimsPredicted", "ultimate_nb_claim_paid_predicted")
    df = _rename_if_exists(df, "UltimateNumberInvoicesPredicted", "ultimate_nb_invoice_paid_predicted")
    return df


@dp.temporary_view
def fall_back_ibnr():

    df = spark.createDataFrame(
        [
            {
                "segment": "no_segment",
                "treatment_date": datetime(1900, 1, 1),
                "full_experiment": "no_experiment",
                "target": "no_target",
                "ultimate_nb_invoice_paid_predicted": -1.0,
                "ultimate_amount_paid_predicted": -1.0,
                "ultimate_nb_claim_paid_predicted": -1,
                "cumulative_nb_claim_paid": -1.0,
                "cumulative_nb_invoice_paid": -1,
                "cumulative_amount_paid": -1.0,
                "MajorICDGroupingDescription": "no_icd",
                "PolicySubType": "no_policy",
            }
        ]
    )

    return df


@dp.temporary_view
def first_experiment():
    df = _normalize_ibnr_columns(spark.table("ibnr_predictions"))
    return df.withColumn("_row_number", F.row_number().over(W.orderBy("full_experiment"))).where(F.col("_row_number") == 1).select("full_experiment")


@dp.temporary_view
def ibnr_predictions():

    default_catalog = spark.conf.get("catalog_name", "no_catalog")
    default_schema = spark.conf.get("schema", "no_schema")

    try:
        df_ibnr = spark.table(f"{default_catalog}.{default_schema}.ibnr_predictions_export_ibnr_inflation")
    except:
        df_ibnr = spark.table("fall_back_ibnr")
    finally:
        df_ibnr = _normalize_ibnr_columns(df_ibnr)
        df_ibnr = df_ibnr.where(F.col("segment") != F.lit("no_segment"))

    return df_ibnr


@dp.materialized_view  # this in theory could be a temprary view, but leaving materialized for simplicity
def ibnr_coefficients():

    df_ibnr = spark.table("ibnr_predictions").join(spark.table("first_experiment"), on="full_experiment", how="inner")

    df_ibnr_coeff = (
        df_ibnr.withColumn("coef_amount", F.col("ultimate_amount_paid_predicted") / F.col("cumulative_amount_paid"))
        .withColumn("coef_nb_claim", F.col("ultimate_nb_claim_paid_predicted") / F.col("cumulative_nb_claim_paid"))
        .withColumn("coef_nb_invoice", F.col("ultimate_nb_invoice_paid_predicted") / F.col("cumulative_nb_invoice_paid"))
    )

    df_ibnr_coeff = df_ibnr_coeff.withColumn("full_experiment", F.regexp_replace("full_experiment", "nb_invoice|nb_claim|Amount|Charge", ""))

    return df_ibnr_coeff
