import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
_feature_store = os.path.normpath(os.path.join(_cwd, "../"))

sys.path.insert(0, _shared)
sys.path.insert(0, _feature_store)

from pyspark import pipelines as dp

from pyspark.sql import functions as F
from utilities.feature_engineering import (
    aggregate_and_process_claims_members_regressors,
)

from inflation_utilities.utils import get_timeframes, get_temporal_column, get_run_config

from shared_utilities.utils import (
    get_config_file,
)


def create_cluster_table(cluster_info):
    @dp.materialized_view(
        name=cluster_info["table_name"],
    )
    def _():
        dataset = dp.read("inflation_aggregated_portfolio")
        regressors = spark.table(get_config_file("inflation", "run_config")["regressors"])

        cluster_id = cluster_info["cluster_id"]
        cluster_filter = clusters.filter(clusters.cluster_id == cluster_id)

        temporal_col = get_temporal_column()

        df = aggregate_and_process_claims_members_regressors(
            df_agg=dataset,
            temporal_col=temporal_col,
            cluster_id=cluster_id,
            cluster_filter=cluster_filter,
            df_regressors=regressors,
        )

        df = df.filter(F.col(temporal_col) >= training_begin_range)

        return df

    return _


training_begin_range = get_timeframes()["training_begin_range"]
clusters_table = get_config_file("inflation", "run_config")["clusters_table"]
clusters = spark.table(clusters_table)

cluster_tables = get_run_config(spark)["cluster_tables"]

for t in cluster_tables.values():
    create_cluster_table(t)
