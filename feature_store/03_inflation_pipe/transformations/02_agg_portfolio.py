import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
_feature_store = os.path.normpath(os.path.join(_cwd, "../"))

sys.path.insert(0, _shared)
sys.path.insert(0, _feature_store)

from pyspark import pipelines as dp

from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime
from shared_utilities.data_quality import (
    add_valid_row_and_dq_flags,
    get_dq_rules,
    get_all_flag_names,
)
from shared_utilities.utils import (
    get_primary_keys,
)
from ibnr_utilities.utils import (
    get_all_segmentation_combinations as get_all_ibnr_segmentation_combinations,
)
from inflation_utilities.utils import (
    get_is_to_compensate_ibnr_on_invoices_view,
)
from utilities.feature_engineering import (
    LOWEST_GRANULARITY_PROPERTIES,
    compensate_ibnr_on_invoices_view,
)

_THIS_TABLE = "inflation_aggregated_portfolio"


@dp.materialized_view(
    name=f"{_THIS_TABLE}_preprocess",
    private=True,
)
def preprocess():

    df_invoice_monthly_agg = spark.table("aggregated_invoices")
    df_contracts_monthly_agg = spark.table("aggregated_contracts")

    if get_is_to_compensate_ibnr_on_invoices_view():
        df_ibnr_coeff = spark.table("ibnr_coefficients")
        segmentation_combinations = get_all_ibnr_segmentation_combinations("ibnr_inflation")
        df_invoice_monthly_agg = compensate_ibnr_on_invoices_view(df_invoice_monthly_agg, df_ibnr_coeff, segmentation_combinations)

    contract_dims = df_contracts_monthly_agg.select("year_month_date", *[key for key, value in LOWEST_GRANULARITY_PROPERTIES.items() if value == "contract"]).distinct()

    invoice_dims = df_invoice_monthly_agg.select(*[key for key, value in LOWEST_GRANULARITY_PROPERTIES.items() if value == "invoice"]).distinct()

    df_all_combinations = contract_dims.crossJoin(invoice_dims)

    df_agg = df_all_combinations.join(df_contracts_monthly_agg, on=contract_dims.columns, how="left")

    df_agg = df_agg.join(df_invoice_monthly_agg, on=contract_dims.columns + invoice_dims.columns, how="left")

    df_agg = df_agg.fillna(
        {
            "nb_invoice": 0,
            "nb_claim": 0,
            "amount_paid": 0,
        }
    )

    df_agg = add_valid_row_and_dq_flags(df_agg, None, get_primary_keys(_THIS_TABLE, "inflation"))

    return df_agg


@dp.materialized_view(
    name=f"{_THIS_TABLE}",
)
def postprocess():
    return spark.table(f"{_THIS_TABLE}_preprocess").filter(F.col("valid_row")).drop(*get_all_flag_names(get_primary_keys(_THIS_TABLE, "inflation")).values(), "valid_row")


@dp.materialized_view(name=f"dq_{_THIS_TABLE}")
@dp.expect_all(get_dq_rules(get_primary_keys(_THIS_TABLE, "inflation")))
def dq():
    return spark.table(f"{_THIS_TABLE}_preprocess").where(~F.col("valid_row")).orderBy(get_primary_keys(_THIS_TABLE, "inflation"))
