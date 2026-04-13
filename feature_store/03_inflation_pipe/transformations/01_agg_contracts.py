import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime
from shared_utilities.utils import (
    get_beginning_of_month_date,
)
from inflation_utilities.utils import (
    get_timeframes,
)
from utilities.feature_engineering import (
    LOWEST_GRANULARITY_PROPERTIES,
)


@dp.temporary_view
def fall_back_contracts():

    df = spark.createDataFrame(
        [
            {
                "PolicyID": "no_policy",
                "VersionStartDate": datetime(1900, 1, 1),
                "VersionEndDate": datetime(1900, 1, 1),
                "DateOfBirth": datetime(1900, 1, 1),
                "member_id": "no_member",
                "PolicySubType": "no_policy",
                "UKMemberRegion": "no_region",
            }
        ]
    )

    return df


@dp.temporary_view
def aggregated_contracts():

    inference_begin_range = get_timeframes()["inference_begin_range"]
    date_to_filter_contracts = get_beginning_of_month_date(inference_begin_range)

    default_catalog = spark.conf.get("catalog_name", "no_catalog")
    default_schema = spark.conf.get("schema", "no_schema")

    try:
        df_contract = spark.table(f"{default_catalog}.{default_schema}.contracts")
    except:
        df_contract = spark.table("fall_back_contracts")
    finally:
        df_contract = df_contract.where(F.col("PolicyID") != F.lit("no_policy"))

    w = Window.partitionBy("PolicyID", "member_id").orderBy(F.col("VersionStartDate").desc())
    df_contract_corrected = df_contract.withColumn("is_latest", F.row_number().over(w) == 1)
    df_contract_corrected = df_contract_corrected.withColumn(
        "VersionEndDate",
        F.when(F.col("is_latest"), F.current_timestamp()).otherwise(F.col("VersionEndDate")),
    ).drop("is_latest")

    df_active_members = (
        spark.table("active_members_preprocess")
        .where(F.col("valid_row"))
        .select(
            "PolicyID",
            "member_id",
            "year_month_date",
        )
    ).withColumnRenamed("YearMonthDate", "year_month_date")

    df_contract_monthly = df_contract_corrected.join(
        df_active_members,
        on=["PolicyID", "member_id"],
        how="inner",
    )
    df_contract_monthly = df_contract_monthly.filter((F.col("VersionStartDate") <= F.col("year_month_date")) & (F.col("VersionEndDate") >= F.col("year_month_date")))

    df_contract_monthly = df_contract_monthly.withColumn(
        "Age",
        F.floor(F.months_between(F.col("year_month_date"), F.col("DateOfBirth")) / 12),
    )
    df_contract_monthly = df_contract_monthly.withColumn(
        "AgeBucket",
        F.when(((0 <= F.col("Age")) & (F.col("Age") <= 20)), F.lit("0_20"))
        .when(((21 <= F.col("Age")) & (F.col("Age") <= 35)), F.lit("21_35"))
        .when(((36 <= F.col("Age")) & (F.col("Age") <= 50)), F.lit("36_50"))
        .when(((51 <= F.col("Age")) & (F.col("Age") <= 60)), F.lit("51_60"))
        .when((60 < F.col("Age")), F.lit("60_plus")),
    ).filter(F.col("AgeBucket").isNotNull())

    df_contract_monthly = df_contract_monthly.filter(
        F.to_date("year_month_date") < date_to_filter_contracts
    )  # WARN : not fully resistant to compute : if inflation code executed after a month to data preprocess

    # Step 2: Calculate total coverage per date
    contract_keys = [key for key, value in LOWEST_GRANULARITY_PROPERTIES.items() if value == "contract"]
    member_id_cols = ["member_id"]
    base_group_cols = ["year_month_date"] + contract_keys

    df_census_policy_agg = df_contract_monthly.groupBy(base_group_cols).agg(F.countDistinct(*member_id_cols).alias("nb_member"))

    from itertools import combinations, chain

    all_combos_to_remove = chain.from_iterable(combinations(contract_keys, r) for r in range(1, len(contract_keys) + 1))

    for removed_keys in all_combos_to_remove:
        removed_keys = list(removed_keys)
        kept_keys = [k for k in contract_keys if k not in removed_keys]

        group_cols = ["year_month_date"] + kept_keys

        df_partial = df_contract_monthly.groupBy(group_cols).agg(F.countDistinct(*member_id_cols).alias("nb_member"))

        for removed_key in removed_keys:
            df_partial = df_partial.withColumn(removed_key, F.lit("_"))

        df_partial = df_partial.select(df_census_policy_agg.columns)
        df_census_policy_agg = df_census_policy_agg.unionByName(df_partial)

    return df_census_policy_agg
