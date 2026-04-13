import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime
from dateutil.relativedelta import relativedelta
from shared_utilities.data_quality import (
    get_all_flag_names,
)
from shared_utilities.utils import (
    get_primary_keys,
    get_beginning_of_month_date,
)
from inflation_utilities.utils import (
    get_timeframes,
)


@dp.materialized_view
def regressors():

    training_end_range = get_timeframes()["training_end_range"]
    expected_month = (training_end_range).strftime("%Y_%m")

    df_external_regressors = (
        spark.table("external_regressors_preprocess").where(F.col("valid_row")).drop(*get_all_flag_names(get_primary_keys("external_regressors", use_case="inflation")).values(), "valid_row")
    ).withColumnRenamed("year_month_date", "year_month_date")

    df_external_regressors = df_external_regressors.filter(F.col("SeenMonth") == expected_month)

    return df_external_regressors
