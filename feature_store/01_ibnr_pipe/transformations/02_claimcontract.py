import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from datetime import datetime
from shared_utilities.utils import (
    get_primary_keys,
)
from shared_utilities.data_quality import (
    add_valid_row_and_dq_flags,
    get_dq_rules,
    get_all_flag_names,
)


@dp.temporary_view
def fall_back_contracts():

    df = spark.createDataFrame(
        [
            {
                "PolicyID": "no_policy",
                "HistoryID": "no_history",
                "MemberID": "no_member",
                "SystemID": "no_systemid",
                "DateOfBirth": datetime(1900, 1, 1),
                "PolicySubType": "no_policy",
                "UKMemberRegion": "no_region",
            }
        ]
    )

    return df


_THIS_TABLE = "claimcontract"


@dp.materialized_view(
    name=f"{_THIS_TABLE}_preprocess",
    private=True,
)
def _():

    default_catalog = spark.conf.get("catalog_name", "no_catalog")
    default_schema = spark.conf.get("schema", "no_schema")

    try:
        df_contract = spark.table(f"{default_catalog}.{default_schema}.contracts")
    except:
        df_contract = spark.table("fall_back_contracts")
    finally:
        df_contract = df_contract.where(F.col("SystemID") != F.lit("no_systemid"))

    df_claim_link = spark.table("claimlink")
    df_claim_contract = df_claim_link.join(
        df_contract.withColumnRenamed("SystemID", "SystemLink"),
        on=["member_id", "SystemLink", "HistoryID", "PolicyID"],
        how="inner",
    )

    df_claim_contract = add_valid_row_and_dq_flags(df_claim_contract, None, get_primary_keys(_THIS_TABLE))

    return df_claim_contract


@dp.materialized_view(
    private=True,
)
def claimcontract():
    return spark.table(f"{_THIS_TABLE}_preprocess").where(F.col("valid_row")).drop(*get_all_flag_names(get_primary_keys(_THIS_TABLE)).values(), "valid_row")


@dp.materialized_view(name=f"dq_{_THIS_TABLE}")
@dp.expect_all(get_dq_rules(get_primary_keys(_THIS_TABLE)))
def _():
    return spark.table(f"{_THIS_TABLE}_preprocess").where(~F.col("valid_row"))
