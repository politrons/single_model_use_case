import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from datetime import date
from ibnr_utilities.utils import (
    get_inference_timeframe_ranges,
)
from shared_utilities.utils import (
    get_primary_keys,
)
from shared_utilities.data_quality import (
    get_all_flag_names,
)

# upper_limit_date = get_cut_off_date()   #### REMOVED AS REQUESTED BY CAMILLE

all_filter_dates = [{"preauth_date": x["preauth_date"], "preauth_date_str": x["preauth_date_str"]} for x in get_inference_timeframe_ranges()]


def create_preauthbase(preauth_date: date, preauth_date_str: str):
    @dp.temporary_view(
        name=f"preauthbase_{preauth_date_str}",
    )
    def _():

        df = spark.table("preauthbase_preprocess").where(F.col("valid_row")).drop(*get_all_flag_names(get_primary_keys("preauthbase")).values(), "valid_row")

        df = (
            df.withColumnRenamed("SystemId", "SystemID")
            .where(F.col("TreatmentEarliestRecord") == "Yes")
            .where(~(F.col("TreatmentDecisionCategory").isin(["Rejected", "Ignore"])))
            .where((F.year("TreatmentEffectiveDate") >= 2008) & (F.col("TreatmentEffectiveDate") < preauth_date))
            .withColumn("TreatmentEffectiveDate" + "_full", F.col("TreatmentEffectiveDate"))
            .withColumn("TreatmentEffectiveDate", F.trunc("TreatmentEffectiveDate", "MM"))
        )

        df = df.withColumnRenamed("ClaimKey", "claim_id")

        return df

    return _


for x in all_filter_dates:
    create_preauthbase(x["preauth_date"], x["preauth_date_str"])
