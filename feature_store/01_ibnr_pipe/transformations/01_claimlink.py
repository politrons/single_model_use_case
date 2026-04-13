from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.temporary_view
def claimlink():

    df = (
        spark.table("claimlink_preprocess")
        .where(F.col("valid_row"))
        .select(
            "ClaimKey",
            "SystemLink",
            "PolicyID",
            "HistoryID",
            "MemberID",
        )
    )

    df = df.withColumnRenamed("ClaimKey", "claim_id").withColumnRenamed("MemberID", "member_id")

    return df
