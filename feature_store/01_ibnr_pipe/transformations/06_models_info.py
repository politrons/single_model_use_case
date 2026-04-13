import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.types import MapType, StringType
from ibnr_utilities.utils import (
    get_run_config,
)

_, _, models_created =  get_run_config(relative_path="../../../")


@dp.materialized_view()
def ibnr_models_info():

    return spark.createDataFrame(models_created).withColumn("model_config", F.from_json(F.col("model_config"), MapType(StringType(), StringType()))).withColumn("ts", F.current_timestamp())
