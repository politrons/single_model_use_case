import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from ibnr_utilities.utils import (
    get_all_targets,
    get_all_run_types,
)


def create_coalesced_table(
    run_type: str,
):
    @dp.materialized_view(
        name=f"ibnr_predictions_coalesced_{run_type}",
    )
    def _():

        df = spark.table("ibnr_predictions_raw").where(F.col(run_type))

        targets = get_all_targets(run_type)
        prediction_cols = [prediction + "_predicted" for prediction in targets]
        first_non_full = [F.first(prediction, ignorenulls=True).alias(prediction) for prediction in prediction_cols]

        df_id_targets = (
            df.select(
                "experiment_without_target",
                "segment_key",
                *prediction_cols,
            )
            .groupby("experiment_without_target", "segment_key")
            .agg(*first_non_full)
        )

        df = df.drop(*prediction_cols).join(df_id_targets, on=["experiment_without_target", "segment_key"], how="left").where(F.col("experiment_without_target") != F.lit("no_experiment"))

        return df

    return _


for run in get_all_run_types():
    create_coalesced_table(run)
