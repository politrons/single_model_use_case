import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

import json
from functools import partial, reduce
from pyspark import pipelines as dp
from pyspark.sql import DataFrame, Window as W
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, TimestampType, BooleanType
from inflation_utilities.utils import (
    get_run_config,
)


def _rename_if_exists(df: DataFrame, source: str, target: str) -> DataFrame:
    if source in df.columns and target not in df.columns:
        return df.withColumnRenamed(source, target)
    return df


def _first_existing(df: DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return F.col(c)
    return None


def _normalize_payload_columns(df: DataFrame) -> DataFrame:
    # Common temporal variants from different payload writers.
    for source in [
        "YearMonthDate",
        "year_month_date",
        "TreatmentDate",
        "treatment_date",
        "ds",
    ]:
        df = _rename_if_exists(df, source, "year_month_date")

    # Common segment variants.
    for source in [
        "cluster_id",
        "ClusterId",
        "segment",
        "Segment",
    ]:
        df = _rename_if_exists(df, source, "cluster_id")

    # Common prediction variants.
    for source in [
        "prediction",
        "Prediction",
        "yhat",
    ]:
        df = _rename_if_exists(df, source, "prediction")

    lower_expr = _first_existing(df, ["prediction_lower", "prediction_interval_lower", "yhat_lower", "PredictionLower"])
    if lower_expr is not None and "prediction_lower" not in df.columns:
        df = df.withColumn("prediction_lower", lower_expr)

    upper_expr = _first_existing(df, ["prediction_upper", "prediction_interval_upper", "yhat_upper", "PredictionUpper"])
    if upper_expr is not None and "prediction_upper" not in df.columns:
        df = df.withColumn("prediction_upper", upper_expr)

    if "year_month_date" in df.columns:
        df = df.withColumn("year_month_date", F.to_date("year_month_date"))
    else:
        df = df.withColumn("year_month_date", F.lit(None).cast(DateType()))

    if "cluster_id" not in df.columns:
        df = df.withColumn("cluster_id", F.lit("_no_cluster"))
    if "prediction" not in df.columns:
        df = df.withColumn("prediction", F.lit(None).cast(DoubleType()))
    if "prediction_lower" not in df.columns:
        df = df.withColumn("prediction_lower", F.lit(None).cast(DoubleType()))
    if "prediction_upper" not in df.columns:
        df = df.withColumn("prediction_upper", F.lit(None).cast(DoubleType()))
    if "is_inference" not in df.columns:
        df = df.withColumn("is_inference", F.lit(True).cast(BooleanType()))
    if "ts" not in df.columns:
        df = df.withColumn("ts", F.current_timestamp())
    else:
        df = df.withColumn("ts", F.col("ts").cast("timestamp"))

    return df


@dp.temporary_view
def fallback_inflation_predictions():
    schema = StructType(
        [
            StructField("cluster_id", StringType(), True),
            StructField("year_month_date", DateType(), True),
            StructField("prediction", DoubleType(), True),
            StructField("prediction_lower", DoubleType(), True),
            StructField("prediction_upper", DoubleType(), True),
            StructField("is_inference", BooleanType(), True),
            StructField("ts", TimestampType(), True),
        ]
    )
    return spark.createDataFrame(
        [
            {
                "cluster_id": "_no_cluster",
                "year_month_date": None,
                "prediction": None,
                "prediction_lower": None,
                "prediction_upper": None,
                "is_inference": True,
                "ts": None,
            }
        ],
        schema=schema,
    )


@dp.materialized_view
def inflation_predictions_raw():
    default_catalog = spark.conf.get("catalog_name", "no_catalog")
    default_schema = spark.conf.get("schema", "no_schema")
    models_created = get_run_config(spark)["models_created"]
    model_infos = list(models_created.values())

    if not model_infos:
        return (
            spark.table("fallback_inflation_predictions")
            .withColumn("model_name", F.lit("no_model"))
            .withColumn("target", F.lit("no_target"))
            .withColumn("origin", F.lit(None).cast("string"))
            .withColumn("model_config", F.lit(None).cast("string"))
        )

    def _read_one_model(model_info: dict) -> DataFrame:
        model_name = model_info["model_name"]
        target = model_info["target"]
        origin = model_info.get("scenario")
        model_config = model_info.get("model_config", {})

        try:
            df_ = spark.table(f"{default_catalog}.{default_schema}.{model_name}_endpoint_payload_flat")
        except Exception:
            df_ = spark.table("fallback_inflation_predictions")

        df_ = _normalize_payload_columns(df_)
        df_ = df_.withColumn("_is_inference", F.coalesce(F.col("is_inference").cast("boolean"), F.lit(True)))
        w_seg = W.partitionBy("cluster_id", "year_month_date").orderBy(F.col("ts").desc_nulls_last())
        df_ = (
            df_
            .where(F.col("_is_inference"))
            .withColumn("_row_number", F.row_number().over(w_seg))
            .where(F.col("_row_number") == 1)
            .where(F.col("year_month_date").isNotNull())
            .select(
                "cluster_id",
                "year_month_date",
                F.col("prediction").cast("double").alias("prediction"),
                F.col("prediction_lower").cast("double").alias("prediction_lower"),
                F.col("prediction_upper").cast("double").alias("prediction_upper"),
                "ts",
            )
            .withColumn("model_name", F.lit(model_name))
            .withColumn("target", F.lit(str(target)))
            .withColumn("origin", F.lit(str(origin) if origin is not None else None))
            .withColumn("model_config", F.lit(json.dumps(model_config)))
        )

        return df_

    dfs = [_read_one_model(model_info) for model_info in model_infos]
    union_by_name_missing = partial(DataFrame.unionByName, allowMissingColumns=True)
    return reduce(union_by_name_missing, dfs)
