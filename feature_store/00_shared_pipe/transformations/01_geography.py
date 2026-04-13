from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.temporary_view
def geography():

    df = (
        spark.table("geography_preprocess")
        .where(F.col("valid_row"))
        .select(
            "PostCode",
            F.col("UkRegion").alias("UKMemberRegion"),
        )
    )

    return df
