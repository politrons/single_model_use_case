import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
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
def fall_back_invoices():

    df = spark.createDataFrame(
        [
            {
                "CareMarkerDescription": "no_caremarkerdescription",
                "treatment_date": datetime(1900, 1, 1),
                "payment_date": datetime(1900, 1, 1),
                "PolicySubType": "no_policy",
                "AgeBucket": "no_age",
                "UKMemberRegion": "no_region",
                "SystemID": "no_system",
                "MajorICDGroupingDescription": "no_icd",
                "amount_paid": -1.0,
                "claim_id": "no_claim",
                "invoice_id": "no_invoice",
            }
        ]
    )

    return df


@dp.temporary_view
def aggregated_invoices():

    inference_begin_range = get_timeframes()["inference_begin_range"]
    date_to_filter_invoices = get_beginning_of_month_date(inference_begin_range)

    default_catalog = spark.conf.get("catalog_name", "no_catalog")
    default_schema = spark.conf.get("schema", "no_schema")

    try:
        df_invoice = spark.table(f"{default_catalog}.{default_schema}.invoices")
    except:
        df_invoice = spark.table("fall_back_invoices")
    finally:
        df_invoice = df_invoice.where(F.col("invoice_id") != F.lit("no_invoice"))

    df_invoice = df_invoice.withColumn("year_month_date", F.trunc("admission_date", "MM"))
    df_invoice = df_invoice.filter(F.col("claim_status").isin("P", "A", "S"))

    df_invoice_monthly_agg = (
        df_invoice.filter((F.col("payment_date").isNotNull()) & (F.col("payment_date") < date_to_filter_invoices))
        .groupBy("year_month_date", *LOWEST_GRANULARITY_PROPERTIES.keys())
        .agg(
            F.sum("amount_paid").alias("amount_paid"),
            F.countDistinct("claim_id").alias("nb_claim"),
            F.count("invoice_id").alias("nb_invoice"),
        )
        .fillna({"nb_invoice": 0, "nb_claim": 0, "amount_paid": 0})
    )

    return df_invoice_monthly_agg
