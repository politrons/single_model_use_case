import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from ibnr_utilities.utils import (
    get_inference_timeframe_ranges,
)
from shared_utilities.data_quality import (
    get_all_flag_names,
)

all_preauth_dates = [x["preauth_date_str"] for x in get_inference_timeframe_ranges()]


def create_preauth(preauth_date_str: str):
    @dp.materialized_view(
        name=f"preauth_{preauth_date_str}",
        private=True,
    )
    def _():

        df_claim_contract = spark.table("claimcontract")

        df_preauth = spark.table(f"preauthbase_{preauth_date_str}")
        df_preauth = df_preauth.join(
            df_claim_contract,
            on=["claim_id"],
            how="inner",
        )

        df_preauth = (
            df_preauth.withColumn("Age", F.floor(F.months_between(F.col("TreatmentEffectiveDate"), F.col("DateOfBirth")) / 12))
            .withColumn(
                "AgeBucket",
                F.when(((0 <= F.col("Age")) & (F.col("Age") <= 20)), F.lit("0_20"))
                .when(((21 <= F.col("Age")) & (F.col("Age") <= 35)), F.lit("21_35"))
                .when(((36 <= F.col("Age")) & (F.col("Age") <= 50)), F.lit("36_50"))
                .when(((51 <= F.col("Age")) & (F.col("Age") <= 60)), F.lit("51_60"))
                .when((60 < F.col("Age")), F.lit("60_plus")),
            )
            .where(F.col("AgeBucket").isNotNull())
            .select(
                "PreAuthTreatmentKey",
                "PreAuthTreatmentHistoryKey",
                "TreatmentEffectiveDate",
                "TreatmentEffectiveDate_full",
                "SystemID",
                "PolicySubType",
                "Age",
                "AgeBucket",
                "UKMemberRegion",
            )
        )

        return df_preauth

    return _


for dt in all_preauth_dates:
    create_preauth(dt)
