import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from functools import reduce
from utilities.feature_engineering import (
    aggregate_and_process_claims_members_regressors,
)
from shared_utilities.utils import (
    get_config_file,
)
from inflation_utilities.utils import (
    get_temporal_column,
)

################################################################################################################################################
# This is not the same DQ as reading the 'raw' source (that is for only  simple uniqueness etc)
# In the original code there 2 addtional checks for regressors:
# [A] - check if there were available with the SeenMonth
# [B] - check if there were available for whole cluster (individually)
# Since when reading regressors they are already filtered by SeeMonth, and for [B] it was done by a 'inner' join, we can merge both checks
# This can be done globally (for all clusters), and just checking if there is any NULL value
################################################################################################################################################


@dp.materialized_view
@dp.expect_or_fail("missing_regressors", "invalid IS NOT TRUE")
def dq_regressors_available():
    dataset = dp.read("inflation_aggregated_portfolio")
    regressors = spark.table(get_config_file("inflation", "run_config")["regressors"])

    temporal_col = get_temporal_column()

    df = aggregate_and_process_claims_members_regressors(
        df_agg=dataset,
        temporal_col=temporal_col,
        cluster_id=None,
        cluster_filter=None,
        df_regressors=regressors,
    )

    df = df.select(*regressors.columns)

    columns_to_check = [c for c in df.columns if c != get_temporal_column()]
    df = df.select(*[F.col(c).isNull().alias(c) for c in columns_to_check], get_temporal_column()).withColumn("invalid", reduce(lambda a, b: a | b, (F.col(c) for c in columns_to_check)))

    return df
