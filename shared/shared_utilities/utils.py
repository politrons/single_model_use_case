import json
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import yaml
from typing import Any


def get_config_file(use_case: str, config_file: str, relative_path: str = "../../../") -> dict:
    import os

    print(os.getcwd())
    with open(f"{relative_path}business_case_config/{use_case}/{config_file}.yml", "r", encoding="utf-8") as yaml_config:
        config = yaml.safe_load(yaml_config)
    return config


def get_full_table_name(table_alias: str, use_case: str = "ibnr") -> str:

    config = get_config_file(use_case, "data_config")

    return config["tables"][table_alias]["full_name"]


def get_primary_keys(table_alias: str, use_case: str = "ibnr") -> list[str]:

    config = get_config_file(use_case, "data_config")

    return config["tables"][table_alias]["primary_keys"]


def _parse_str_to_date(this_date: str | date | datetime) -> date:
    match this_date:
        case datetime() as this_datetime:
            return this_datetime.date()
        case date() as this_date_actual:
            return this_date_actual
        case str() as this_date_str:
            return datetime.strptime(this_date_str.replace("_", "-"), "%Y-%m-%d").date()
        case _:
            raise TypeError("format not recognised")


def get_beginning_of_month_date(this_date: str | date | datetime) -> date:

    parsed_date = _parse_str_to_date(this_date)

    first_day_date_obj = parsed_date.replace(day=1)

    return first_day_date_obj


def first_of_following_month(this_date: str | date | datetime) -> date:

    parsed_date = _parse_str_to_date(this_date)

    return (parsed_date + relativedelta(months=1)).replace(day=1)


def _parse_str_to_bool(this_bool: str | bool) -> bool:

    true_values = ["True", "true", True, 1]
    false_values = ["False", "false", False, 0]

    if this_bool in true_values:
        return True
    elif this_bool in false_values:
        return False
    else:
        raise TypeError("format not recognised")
