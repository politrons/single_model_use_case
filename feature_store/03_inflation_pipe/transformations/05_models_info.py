import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
_feature_store = os.path.normpath(os.path.join(_cwd, "../"))

sys.path.insert(0, _shared)
sys.path.insert(0, _feature_store)

from pyspark import pipelines as dp

import json

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from inflation_utilities.utils import (
    get_run_config,
)

models_created = get_run_config(spark)["models_created"]


@dp.materialized_view()
def inflation_models_info():

    schema = StructType(
        [
            StructField("model_name", StringType(), nullable=False),
            StructField("target", StringType(), nullable=False),
            StructField("scenario", StringType(), nullable=False),
            StructField("table_name", StringType(), nullable=False),
            StructField("model_config", ArrayType(ArrayType(StringType())), nullable=False),
        ]
    )

    parsed = [
        {
            "model_name": model_info["model_name"],
            "target": model_info["target"],
            "scenario": model_info["scenario"],
            "table_name": model_info["table_name"],
            "model_config": [[cluster_id, json.dumps(config)] for cluster_id, config in model_info["model_config"].items()],
        }
        for model_info in list(models_created.values())
    ]

    return spark.createDataFrame(parsed, schema=schema).withColumn("ts", F.current_timestamp())
