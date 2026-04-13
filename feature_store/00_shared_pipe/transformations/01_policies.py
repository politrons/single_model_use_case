import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from ibnr_utilities.utils import get_min_available_timeframe


@dp.temporary_view
def policies():

    df = (
        spark.table("policies_preprocess")
        .where(F.col("valid_row"))
        .select(
            "PolicyID",
            "SystemID",
            "HistoryID",
            "PolicySubType",
            "PolicyType",
            "ProductFamily",
            "CoverStartDate",
            "RenewalDate",
        )
    )

    filtered_policy_types_expr = ~(
        F.col("PolicyType").isin(
            [
                "SAGA",
                "SAGA Accidental Healthcare",
                "Dental",
                "Cashback",
            ]
        )
    )

    filtered_product_families = [
        "JERSEY",
        "GUERNSEY",
        "IHP",
    ]
    filtered_product_families_expr = ~(F.col("ProductFamily").isin(filtered_product_families) | F.col("ProductFamily").contains("IHP") | F.col("ProductFamily").contains("SAGA")) | (
        F.col("ProductFamily").isNull()
    )

    filtered_policy_subtypes_expr = F.col("PolicySubType").isin(
        [
            "Employee (LC)",
            "Employee (SME)",
            "Individual",
            "Trust",
        ]
    )

    df = (
        df.withColumn(
            "PolicySubType",
            F.when(F.col("PolicyType") != "Trust", F.col("PolicySubType")).otherwise(F.lit("Trust")),
        )
        .where(filtered_policy_types_expr)
        .where(filtered_product_families_expr)
        .where(filtered_policy_subtypes_expr)
        .where(F.to_date("CoverStartDate") >= get_min_available_timeframe())
    )

    return df
