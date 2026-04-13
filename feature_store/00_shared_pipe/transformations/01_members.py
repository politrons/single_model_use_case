from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.temporary_view
def members():

    df = (
        spark.table("members_preprocess")
        .where(F.col("valid_row"))
        .select(
            "MemberID",
            "PolicyID",
            "SystemID",
            "HistoryID",
            "PolicyHistoryID",
            "PostCode",
            "VersionStartDate",
            "VersionEndDate",
            "CancellationDate",
            "DateOfBirth",
            "Gender",
        )
    )

    df = df.withColumn(
        "PostCode",
        F.split(F.col("PostCode"), " ")[0],
    )

    df = df.withColumnRenamed("MemberID", "member_id")

    return df
