import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from shared_utilities.utils import get_full_table_name, get_primary_keys
from shared_utilities.data_quality import (
    add_valid_row_and_dq_flags,
    get_dq_count,
    get_dq_rules,
)

_ALL_TABLES = [
    "policies",
    "members",
    "geography",
    "invoicebase",
    "invoicelink",
]


def create_preprocess(table_name: str):
    @dp.materialized_view(
        name=f"{table_name}_preprocess",
    )
    def _():
        return add_valid_row_and_dq_flags(spark, get_full_table_name(table_name), get_primary_keys(table_name))

    return _


def create_grouped_dq(table_name: str):
    @dp.materialized_view(name=f"dq_{table_name}")
    @dp.expect_all(get_dq_rules(get_primary_keys(table_name)))
    def _():
        return spark.table(f"{table_name}_preprocess").where(~F.col("valid_row"))

    return _


for t in _ALL_TABLES:
    create_preprocess(t)
    create_grouped_dq(t)
