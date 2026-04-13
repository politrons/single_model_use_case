import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from ibnr_utilities.utils import (
    get_maximal_years_runoff_lags,
    get_run_config,
)
from utilities.feature_engineering import (
    maximal_feature_engineering,
)

maximal_years_runoff_lags = get_maximal_years_runoff_lags()


def create_aggregated_table(
    table_name: str,
    this_segmentation: list[str],
    preauth_date_str: str,
    max_lags: int,
):
    @dp.materialized_view(
        name=f"{table_name}",
    )
    def _():

        df_invoice = spark.table("invoices_ibnr")
        df_preauth = spark.table(f"preauth_{preauth_date_str}")

        df_agg = maximal_feature_engineering(
            df_invoice=df_invoice,
            df_preauth=df_preauth,
            segmentation_features=this_segmentation,
            max_lags=max_lags,
            maximal_years_runoff_lags=maximal_years_runoff_lags,
        )

        return df_agg

    return _


aggregated_tables, _, _ = get_run_config()

for agg in aggregated_tables.values():
    create_aggregated_table(table_name=agg["table_name"], this_segmentation=agg["segmentation_combination"], preauth_date_str=agg["preauth_date_str"], max_lags=agg["max_lags"])
