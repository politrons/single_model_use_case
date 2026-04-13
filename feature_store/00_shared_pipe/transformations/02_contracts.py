import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from shared_utilities.utils import get_primary_keys
from shared_utilities.data_quality import (
    add_valid_row_and_dq_flags,
    get_dq_rules,
    get_all_flag_names,
)

_THIS_TABLE = "contracts"


@dp.materialized_view(
    name=f"{_THIS_TABLE}_preprocess",
    private=True,
)
def _():

    df_policy = spark.table("policies")
    df_member = spark.table("members")
    df_geography = spark.table("geography")

    df = df_member.join(
        df_geography,
        on=["PostCode"],
        how="left",
    ).fillna({"UKMemberRegion": "Unknown"})

    df = df_policy.withColumnRenamed("HistoryID", "PolicyHistoryID").join(
        df,
        on=["PolicyID", "SystemID", "PolicyHistoryID"],
        how="inner",
    )

    df = add_valid_row_and_dq_flags(df, None, get_primary_keys(_THIS_TABLE))

    return df


@dp.materialized_view(
    name=f"{_THIS_TABLE}",
)
def _():
    return spark.table(f"{_THIS_TABLE}_preprocess").where(F.col("valid_row")).drop(*get_all_flag_names(get_primary_keys(_THIS_TABLE)).values(), "valid_row")


@dp.materialized_view(name=f"dq_{_THIS_TABLE}")
@dp.expect_all(get_dq_rules(get_primary_keys(_THIS_TABLE)))
def _():
    return spark.table(f"{_THIS_TABLE}_preprocess").where(~F.col("valid_row"))
