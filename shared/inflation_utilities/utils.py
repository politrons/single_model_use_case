import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../shared"))
sys.path.insert(0, _shared)

from typing import Any
from pyspark.sql import functions as F
from pyspark.sql import Window as W
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import MapType, StringType
import json
import re
import hashlib
import base64
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from collections import defaultdict
from shared_utilities.utils import (
    get_config_file,
    _parse_str_to_bool,
    _parse_str_to_date,
)


def sanitize_dbx_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    return sanitized.lower()


def short_id(s, length=6):
    digest = hashlib.sha256(s.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode()[:length]


def sanitize_table_name(name: str) -> str:
    if len(name) > 255:
        raise ValueError("Databricks/UC only allow table names up to 255 chars")

    return sanitize_dbx_name(name)


def get_is_to_compensate_ibnr_on_invoices_view() -> bool:

    config = get_config_file("inflation", "run_config")

    return _parse_str_to_bool(config["is_to_compensate_ibnr_on_invoices_view"])


def get_timeframes(relative_path: str = "../../../") -> dict[str, Any]:

    config = get_config_file("inflation", "run_config", relative_path)

    timeframe = config["timeframe_info"]

    use_custom_ranges = timeframe["use_custom_ranges"]

    if use_custom_ranges:
        nb_forecast_years = timeframe["nb_forecast_years"]
        training_begin_range = _parse_str_to_date(timeframe["training_begin"])
        training_end_range = _parse_str_to_date(timeframe["training_end"])
        inference_begin_range = (training_end_range + relativedelta(months=1)).replace(day=1)
        inference_end_range = (inference_begin_range + relativedelta(years=nb_forecast_years) - relativedelta(months=1)).replace(day=1)
    else:
        input_date = datetime.today()
        nb_forecast_years = 3

        inference_begin_range = (input_date).replace(day=1).date()
        inference_end_range = (input_date + relativedelta(years=nb_forecast_years) - relativedelta(months=1)).replace(day=1).date()
        training_begin_range = date(2016, 1, 1)
        training_end_range = (input_date - relativedelta(months=1)).replace(day=1).date()

    all_timeframes = {
        "inference_begin_range": inference_begin_range,
        "inference_end_range": inference_end_range,
        "training_begin_range": training_begin_range,
        "training_end_range": training_end_range,
    }

    return all_timeframes


def get_model_name(prefix: str, table_name: str, target: str, scenario: str) -> tuple[str, str]:

    model_base_name = table_name.replace(f"{prefix}_gold", "inflation")

    model_full_name = model_base_name + f"_scenario_{scenario}" + f"_target_{target}"

    model_name = f"{prefix}_{short_id(model_full_name)}"

    model_name = sanitize_dbx_name(model_name)

    assert len(model_name) < 20

    return model_name, model_full_name


def get_run_config(spark, relative_path: str = "../../../") -> dict:

    config = get_config_file("inflation", "run_config", relative_path)
    clusters_table = config["clusters_table"]
    df = spark.table(clusters_table)

    cluster_id_column = "cluster_id"
    model_config_column = "model_config"
    scenario_column = "scenario"
    target_column = "target"

    gold_table_name = "inflation_gold"
    cluster_tables = {}
    gold_tables = {}
    models_created = {}

    unique_cluster_ids = [
        row.cluster_id 
        for row in df.select("cluster_id").distinct().collect()
    ]
    for this_cluster_id in unique_cluster_ids:
        short_cluster_id = short_id(this_cluster_id)
        cluster_table_name = f"inflation_cluster_{short_cluster_id}"
        cluster_table_name = sanitize_table_name(cluster_table_name)
        cluster_tables[cluster_table_name] = {"table_name": cluster_table_name, "cluster_id": this_cluster_id}

    gold_table_name = "inflation_gold"
    gold_tables[gold_table_name] = {
        "table_name": gold_table_name,
        "clusters": list(cluster_tables.keys()),
    }

    combinations = df.select(target_column, scenario_column).distinct().collect()

    for row in combinations:
        this_target = row[0]
        this_scenario = row[1]
        model_name, model_full_name = get_model_name(prefix="inflation", table_name=gold_table_name, target=this_target, scenario=this_scenario)

        result = df.filter(F.col(target_column) == this_target).filter(F.col(scenario_column) == this_scenario).select(cluster_id_column, model_config_column).collect()

        cluster_configs = {row[cluster_id_column]: row[model_config_column] for row in result}

        models_created[model_name] = {"model_name": model_name, "scenario": this_scenario, "target": this_target, "table_name": gold_table_name, "model_config": cluster_configs}

    run = {}
    run["cluster_tables"] = cluster_tables
    run["gold_tables"] = gold_tables
    run["models_created"] = models_created

    return run


def get_temporal_column(relative_path: str = "../../../") -> str:

    config = get_config_file("inflation", "feature_engineering_config", relative_path)

    return config["temporal_column"]
