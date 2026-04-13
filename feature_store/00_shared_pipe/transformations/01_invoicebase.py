import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from datetime import datetime

from ibnr_utilities.utils import get_min_available_timeframe
from shared_utilities.utils import get_primary_keys
from shared_utilities.data_quality import (
    add_valid_row_and_dq_flags,
    get_dq_rules,
    get_all_flag_names,
)


@dp.temporary_view
def invoicebase():

    df = (
        spark.table("invoicebase_preprocess")
        .where(F.col("valid_row"))
        .select(
            "InvoiceKey",
            "InvoiceID",
            "SystemID",
            "System",
            "TreatmentDate",
            "DateReceived",
            "PaidDate",
            "AmountPaid",
            "MajorICDGroupingDescription",
            "InvoiceStatus",
            "CareMarkerDescription",
        )
    )

    df = (
        df.withColumnRenamed("TreatmentDate", "treatment_date")
        .withColumnRenamed("PaidDate", "payment_date")
        .withColumnRenamed("DateReceived", "reception_date")
        .withColumnRenamed("AmountPaid", "amount_paid")
        .withColumnRenamed("InvoiceKey", "invoice_id")
    )

    df = (
        df.fillna({"MajorICDGroupingDescription": "Unknown"})
        .where(F.col("System").isin(["ACS", "CIMA"]))
        .where(F.col("InvoiceStatus").isin(["Paid", "Pending", "Cancelled"]))
        .where(F.col("treatment_date").isNotNull())
        .where((F.to_date("treatment_date") >= get_min_available_timeframe()) & (F.to_date("treatment_date") < datetime.today().replace(day=1).strftime("%Y-%m-%d")))
        .where((F.col("payment_date").isNull()) | (F.col("treatment_date") <= F.col("payment_date")))
    )

    return df
