import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from datetime import datetime
from ibnr_utilities.utils import get_occured_date_col
from shared_utilities.utils import get_primary_keys


@dp.temporary_view
def fall_back_invoices():

    df = spark.createDataFrame(
        [
            {
                "reception_date": datetime(1900, 1, 1),
                get_occured_date_col(): datetime(1900, 1, 1),
                "payment_date": datetime(1900, 1, 1),
                "invoice_id": "no_invoice",
                "PolicySubType": "no_policy",
                "AgeBucket": "no_bucket",
                "UKMemberRegion": "no_region",
                "Age": -1,
                "SystemID": "no_system",
                "MajorICDGroupingDescription": "no_icd",
                "amount_paid": -1.0,
                "claim_id": "no_claim",
            }
        ]
    )

    return df


@dp.materialized_view
def invoices_ibnr():

    default_catalog = spark.conf.get("catalog_name", "no_catalog")
    default_schema = spark.conf.get("schema", "no_schema")

    try:
        df = spark.table(f"{default_catalog}.{default_schema}.invoices")
    except:
        df = spark.table("fall_back_invoices")
    finally:
        df = df.where(F.col("invoice_id") != F.lit("no_invoice"))

    occured_date_col = get_occured_date_col()
    received_date_col = "reception_date"
    paid_date_col = "payment_date"

    df = (
        df.withColumn("daily_lag_treatment_reception", F.date_diff(received_date_col, occured_date_col).cast("int"))
        .fillna({"daily_lag_treatment_reception": 99 * 365})
        .withColumn("daily_lag_treatment_payment", F.date_diff(paid_date_col, occured_date_col).cast("int"))
        .fillna({"daily_lag_treatment_payment": 99 * 365})
        .withColumn("daily_lag_reception_payment", F.date_diff(paid_date_col, received_date_col).cast("int"))
        .fillna({"daily_lag_reception_payment": 99 * 365})
    )

    if occured_date_col + "_full" not in df.columns:
        df = (
            df.withColumn(occured_date_col + "_full", F.col(occured_date_col))
            .withColumn(occured_date_col, F.trunc(occured_date_col, "MM"))
            .withColumn(received_date_col + "_full", F.col(received_date_col))
            .withColumn(received_date_col, F.trunc(received_date_col, "MM"))
            .withColumn(paid_date_col + "_full", F.col(paid_date_col))
            .withColumn(paid_date_col, F.trunc(paid_date_col, "MM"))
        )

    df = (
        df.withColumn(occured_date_col + "_month", F.month(F.to_date(occured_date_col)))
        .withColumn(received_date_col + "_month", F.month(F.to_date(received_date_col)))
        .withColumn("lag_treatment_reception", F.months_between(received_date_col, occured_date_col).cast("int"))
        .fillna({"lag_treatment_reception": 99})
        .withColumn("lag_treatment_payment", F.months_between(paid_date_col, occured_date_col).cast("int"))
        .fillna({"lag_treatment_payment": 99})
        .withColumn("lag_reception_payment", F.months_between(paid_date_col, received_date_col).cast("int"))
        .fillna({"lag_reception_payment": 99})
    )

    return df
