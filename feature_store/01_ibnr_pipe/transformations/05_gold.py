import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from datetime import date
from dateutil.relativedelta import relativedelta
from ibnr_utilities.utils import (
    get_occured_date_col,
    get_run_config,
)


def create_gold_table(
    table_name: str,
    seg_cols: list[str],
    training_years: int,
    aggregated_table_name: str,
    this_start_date: date,
    this_end_date: date,
):
    @dp.materialized_view(
        name=table_name,
    )
    def _():
        df_gold = spark.table(aggregated_table_name)
        OCCURED_DATE_COL = get_occured_date_col()

        ######################################## modified by gustavo.martins
        # slighty changing the filtering,
        # because the framework will do the splitting.
        # and the filtering
        df_gold = df_gold.where((F.col(OCCURED_DATE_COL) >= this_start_date - relativedelta(years=training_years)) & (F.col(OCCURED_DATE_COL) <= this_end_date))
        ######################################## end

        df_gold = df_gold.withColumn("segment", F.concat_ws("_", *[F.col(c) for c in seg_cols])).withColumn("segment_key", F.concat_ws("_", *[F.col(c) for c in seg_cols + ["lag"]]))

        ######################################## modified by gustavo.martins
        inference_cond = F.col("lag") == ((this_end_date.year - F.year(OCCURED_DATE_COL)) * 12 + this_end_date.month - F.month(OCCURED_DATE_COL))
        inference_cond = inference_cond & (F.col(OCCURED_DATE_COL) >= this_start_date)

        df_gold = df_gold.withColumn("is_inference", inference_cond)
        ######################################## end

        return df_gold

    return _


_, gold_tables, _ = get_run_config(relative_path="../../../")

for gold in gold_tables.values():
    create_gold_table(
        table_name=gold["table_name"],
        seg_cols=gold["segmentation_combination"],
        training_years=gold["training_years"],
        aggregated_table_name=gold["aggregated_table_name"],
        this_start_date=gold["start_date"],
        this_end_date=gold["end_date"],
    )
