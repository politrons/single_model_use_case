import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from datetime import datetime


@dp.temporary_view
def invoicelink():

    df = (
        spark.table("invoicelink_preprocess")
        .where(F.col("valid_row"))
        .select(
            "InvoiceKey",
            "ClaimKey",
            "SystemLink",
            "MemberID",
            "HistoryID",
            "PolicyID",
        )
    )

    df = df.withColumnRenamed("MemberID", "member_id")
    df = df.withColumn("claim_id", F.coalesce(F.col("ClaimKey"), F.col("member_id")))
    df = df.withColumnRenamed("InvoiceKey", "invoice_id")

    return df
