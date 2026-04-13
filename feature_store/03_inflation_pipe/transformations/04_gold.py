import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
_feature_store = os.path.normpath(os.path.join(_cwd, "../"))

sys.path.insert(0, _shared)
sys.path.insert(0, _feature_store)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from functools import reduce, partial
from inflation_utilities.utils import (
    get_run_config,
)


def create_gold_table(gold_info):
    @dp.materialized_view(
        name=gold_info["table_name"],
    )
    def _():

        dfs = [spark.table(cluster_table_name) for cluster_table_name in gold_info["clusters"]]

        union_by_name_missing = partial(DataFrame.unionByName, allowMissingColumns=True)
        df = reduce(union_by_name_missing, dfs)

        return df

    return _


gold_tables = get_run_config(spark)["gold_tables"]

for t in gold_tables.values():
    create_gold_table(t)
