import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql import Window as W
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, StringType, MapType
from functools import reduce, partial
from datetime import datetime
from ibnr_utilities.utils import (
    get_run_config,
)

_, _, models_created = get_run_config()


@dp.temporary_view
def fall_back_ibnr_predictions():

    df = spark.createDataFrame(
        [
            {
                "segment_key": "no_segment_key",
                "segment": "no_segment",
                "ultimate_nb_invoice_paid_predicted": -1,
                "ultimate_amount_paid_predicted": -1.0,
                "ultimate_nb_claim_paid_predicted": -1.0,
                "cumulative_nb_invoice_paid": -1,
                "cumulative_amount_paid": -1.0,
                "cumulative_nb_claim_paid": -1,
                "treatment_date": datetime(1900, 1, 1),
                "lag": -1,
                "PolicySubType": "no_record",
                "SystemID": "no_record",
                "MajorICDGroupingDescription": "no_record",
                "ts": datetime(1900, 1, 1),
                "is_inference": False,
            }
        ]
    )

    return df


@dp.materialized_view()
def ibnr_predictions_raw():

    default_catalog = spark.conf.get("catalog_name", "no_catalog")
    default_schema = spark.conf.get("schema", "no_schema")

    df_models_info = spark.createDataFrame(models_created).withColumn("model_config", F.from_json(F.col("model_config"), MapType(StringType(), StringType())))

    def read_predictions_dedup_filter(model_info: dict) -> DataFrame:

        model_name = model_info["model_name"]
        segmentation_columns_with_possible_lag = model_info["segmentation_columns_with_possible_lag"]
        target = model_info["target"]

        ### read predictions with possible fall back
        try:
            df_ = spark.table(f"{default_catalog}.{default_schema}.{model_name}_endpoint_payload_flat")
        except:
            df_ = spark.table("fall_back_ibnr_predictions").withColumn("model_name", F.lit(model_name))

        ### filter latest predictions by model
        w_seg = W.partitionBy(*segmentation_columns_with_possible_lag).orderBy(F.col("ts").desc())
        df_ = (
            df_.where(F.col("is_inference"))
            .withColumn("_row_number", F.row_number().over(w_seg))
            .where(F.col("_row_number") == 1)
            .withColumn("model_name", F.lit(model_name))
            .join(df_models_info, on="model_name", how="inner")
        )

        ### cosmectic parse
        df_ = df_.withColumnRenamed("prediction", target + "_predicted").withColumn("treatment_date", F.col("treatment_date").cast("date"))

        return df_

    dfs = [read_predictions_dedup_filter(v) for v in models_created]

    union_by_name_missing = partial(DataFrame.unionByName, allowMissingColumns=True)
    df = reduce(union_by_name_missing, dfs)

    ### safety filter
    w_dummy = W.partitionBy(F.lit(1))
    df = (
        df.withColumn("latest_prediction", F.max("ts").over(w_dummy))
        .withColumn("since_latest_prediction_hours", (F.col("latest_prediction").cast("long") - F.col("ts").cast("long")) / 3600)
        .where(F.col("since_latest_prediction_hours") < 8)
        .drop(
            "_row_number",
            "prediction_proba",
            "model_id_col",
            "label_col",
            "feature_combination",
            "model_config",
            "model_full_name",
            "segmentation",
            "table_name",
            "training_years",
            "is_per_lag_modeling",
            "ts",
            "is_inference",
            "job_compute",
            "model_type",
            "latest_prediction",
            "since_latest_prediction_hours",
        )
    )

    return df
