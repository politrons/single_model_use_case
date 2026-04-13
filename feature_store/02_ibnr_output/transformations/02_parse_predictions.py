import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql import Window as W
from pyspark.sql import DataFrame
from functools import reduce
from ibnr_utilities.utils import (
    get_all_targets,
    get_all_run_types,
)

prices_smoothed = {}

for run_type in get_all_run_types():
    targets = get_all_targets(run_type)

    prices_smoothed[run_type] = {
        c: {
            "first_experiment": f"{run_type}_first_full_exp_{c}",
            "avg_n_invoices_to_amount": f"{run_type}_avg_n_invoices_to_amount_{c}",
            "cumulative_col": c.replace("ultimate", "cumulative").replace("_predicted", ""),
            "price_smoothed_column": f"price_smoothed_{c}",
            "run_type": run_type,
        }
        for c in targets
        if "ultimate_nb" in c
    }


def get_first_full_exp(c: str, info: dict):
    @dp.materialized_view(
        name=info["first_experiment"],
        private=True,
    )
    def _():
        cumulative_col = info["cumulative_col"]
        return (
            spark.table(f"ibnr_predictions_coalesced_{info['run_type']}")
            .where(F.col(cumulative_col).isNotNull() & F.col("full_experiment").contains(c))
            .withColumn("_row_number", F.row_number().over(W.orderBy("full_experiment")))
            .where(F.col("_row_number") == 1)
            .select("full_experiment")
        )

    return _


def create_df_avg_n_invoices_to_amount(c: str, info: dict):
    @dp.materialized_view(
        name=info["avg_n_invoices_to_amount"],
        private=True,
    )
    def _():

        cumulative_col = info["cumulative_col"]

        first_full_exp = spark.table(info["first_experiment"])

        df_avg_n_invoices_to_amount = spark.table(f"ibnr_predictions_coalesced_{info['run_type']}").join(first_full_exp, on="full_experiment", how="inner")

        df_avg_n_invoices_to_amount = df_avg_n_invoices_to_amount.withColumn("price", F.col("cumulative_amount_paid") / F.col(cumulative_col))

        w_center = W.partitionBy("segment", "target").orderBy("treatment_date").rowsBetween(-2, 2)

        df_avg_n_invoices_to_amount = df_avg_n_invoices_to_amount.orderBy(["segment", "treatment_date", "target"]).withColumn("price_smoothed", F.avg("price").over(w_center))

        df_avg_n_invoices_to_amount = (
            df_avg_n_invoices_to_amount.where(F.col("lag") > 4)
            .select("segment", "treatment_date", "target", "lag", "price_smoothed")
            .withColumnRenamed("price_smoothed", info["price_smoothed_column"])
        )

        return df_avg_n_invoices_to_amount

    return _


def create_export(run_type: str):
    @dp.materialized_view(
        name=f"ibnr_predictions_export_{run_type}",
    )
    def _():

        df_all_preds = spark.table(f"ibnr_predictions_coalesced_{run_type}")

        def join_one_source(current: DataFrame, c: str) -> DataFrame:
            price_smoothed = prices_smoothed[run_type][c]["price_smoothed_column"]

            source_df = spark.table(prices_smoothed[run_type][c]["avg_n_invoices_to_amount"]).select(*["segment", "treatment_date", "lag", "target"], F.col(price_smoothed))

            return current.join(source_df, on=["segment", "treatment_date", "lag", "target"], how="left")

        df_all_preds = reduce(join_one_source, prices_smoothed[run_type].keys(), df_all_preds)

        all_price_smoothed_columns = [F.col(info["price_smoothed_column"]) for info in prices_smoothed[run_type].values()]

        df_all_preds = df_all_preds.withColumn("price_smoothed", F.coalesce(*all_price_smoothed_columns)).drop(*all_price_smoothed_columns)

        df_all_preds = df_all_preds.withColumn("VisibleDate", F.add_months(F.col("treatment_date"), F.col("lag").cast("int")))

        # Sort by segment (A→Z), VisibleDate (old→new), lag (large→small)
        df_all_preds = df_all_preds.orderBy(F.col("segment").asc(), F.col("VisibleDate").asc(), F.col("lag").desc())

        w_ffill = W.partitionBy("segment", "VisibleDate", "target").orderBy(F.desc("lag")).rowsBetween(W.unboundedPreceding, W.currentRow)
        df_all_preds = df_all_preds.withColumn("price_smoothed", F.last("price_smoothed", ignorenulls=True).over(w_ffill))

        # fallback for first run
        if "ultimate_amount_paid_predicted" not in df_all_preds.columns:
            df_all_preds = df_all_preds.withColumn("ultimate_amount_paid_predicted", F.lit(None))

        for c in prices_smoothed[run_type].keys():
            predicted_col = c + "_predicted"
            df_all_preds = df_all_preds.withColumn(
                "ultimate_amount_paid_predicted", F.when(F.col("target") == c, F.col(predicted_col) * F.col("price_smoothed")).otherwise(F.col("ultimate_amount_paid_predicted"))
            )

        return df_all_preds

    return _


for run_type in prices_smoothed.keys():
    for c, info in prices_smoothed[run_type].items():
        get_first_full_exp(c, info)
        create_df_avg_n_invoices_to_amount(c, info)
    create_export(run_type)
