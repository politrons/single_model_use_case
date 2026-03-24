import logging
from typing import Any, Tuple

import pandas as pd   # type: ignore # noqa
from pyspark.sql import DataFrame, SparkSession   # type: ignore # noqa

from databricks_mlops_stack.utils.spark_table_data_loader import (   # type: ignore # noqa
    SparkTableDataLoader,
    get_active_or_create_spark,
)

# ----------------------------- logging -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
LOG = logging.getLogger("framework.training.data")

# --- seam to obtain a SparkSession (easy to monkeypatch in tests) ---
def _get_spark() -> SparkSession:
    """Return the active Spark session or create one.
    On Databricks, the global session is already active; locally we create one.
    Tests will monkeypatch this function to return a Fake/Mock Spark.
    """
    return get_active_or_create_spark()

class TrainingDataConfig:

    def get_training_data(self, args: dict[str, Any]) -> Tuple[pd.DataFrame | DataFrame, pd.Series | DataFrame]:
        """Retrieve training data (features + target) from config using shared Spark loader."""
        loader = SparkTableDataLoader(spark_getter=_get_spark)
        return loader.load_xy(args)

build = TrainingDataConfig()
