import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from shared_utilities.utils import get_primary_keys
from shared_utilities.data_quality import (
    get_all_flag_names,
    add_valid_row_and_dq_flags,
    get_dq_rules,
)

_THIS_TABLE = "invoices"


@dp.materialized_view(
    name=f"{_THIS_TABLE}_preprocess",
    private=True,
)
def invoices():

    df_invoice = spark.table("invoicebase")
    df_invoice_link = spark.table("invoicelink")
    df_policy_member = spark.table("contracts").drop(*get_all_flag_names(get_primary_keys("contracts")).values(), "valid_row")

    # Merge invoice link with policy/contract
    df_policy_member_invoice = df_invoice_link.join(
        df_policy_member.withColumnRenamed("SystemID", "SystemLink"),
        on=["member_id", "SystemLink", "HistoryID", "PolicyID"],
        how="left",
    )

    # Merge invoice with policy/contract data
    df_invoice = df_invoice.join(
        df_policy_member_invoice.select("invoice_id", "claim_id", "PolicySubType", "DateOfBirth", "UKMemberRegion"),
        on=["invoice_id"],
        how="inner",
    )

    # Add features
    df_invoice = (
        df_invoice.withColumn(
            "Age",
            F.floor(F.months_between(F.col("treatment_date"), F.col("DateOfBirth")) / 12),
        )
        .withColumn(
            "AgeBucket",
            F.when(((0 <= F.col("Age")) & (F.col("Age") <= 20)), F.lit("0_20"))
            .when(((21 <= F.col("Age")) & (F.col("Age") <= 35)), F.lit("21_35"))
            .when(((36 <= F.col("Age")) & (F.col("Age") <= 50)), F.lit("36_50"))
            .when(((51 <= F.col("Age")) & (F.col("Age") <= 60)), F.lit("51_60"))
            .when((60 < F.col("Age")), F.lit("60_plus")),
        )
        .where(F.col("AgeBucket").isNotNull())
    )

    ######################################## modified by gustavo.martins
    # add DQs because in inflation this is verified
    df_invoice = add_valid_row_and_dq_flags(df_invoice, None, get_primary_keys(_THIS_TABLE))
    ######################################## end

    return df_invoice


@dp.materialized_view(
    name=f"{_THIS_TABLE}",
)
def _():
    return spark.table(f"{_THIS_TABLE}_preprocess").where(F.col("valid_row")).drop(*get_all_flag_names(get_primary_keys(_THIS_TABLE)).values(), "valid_row")


@dp.materialized_view(name=f"dq_{_THIS_TABLE}")
@dp.expect_all(get_dq_rules(get_primary_keys(_THIS_TABLE)))
def _():
    return spark.table(f"{_THIS_TABLE}_preprocess").where(~F.col("valid_row"))
