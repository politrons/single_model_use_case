import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../shared"))
sys.path.insert(0, _shared)

import re
import hashlib
import base64
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from typing import Any, Optional
from collections import defaultdict
from shared_utilities.utils import get_config_file, _parse_str_to_date, get_beginning_of_month_date, first_of_following_month


def sanitize_dbx_name(name: str) -> str:

    sanitized = re.sub(r"[.\s/()[\]{}|]|[\x00-\x1F\x7F]", "_", name)
    sanitized = sanitized.replace("-", "_")

    # Force lowercase (Unity Catalog stores names in lowercase)
    return sanitized.lower()


def sanitize_table_name(name: str) -> str:
    if len(name) > 255:
        raise ValueError("Databricks/UC only allow table names up to 255 chars")

    return sanitize_dbx_name(name)


def get_gold_table_name(
    segmentation: list[str],
    training_years: int,
    start_date_str: str,
    end_date_str: str,
) -> str:

    name_parts = [f"{col}" for col in segmentation]
    table_name = "ibnr_gold"
    table_name += f"__years_{training_years}"
    table_name += f"__start_{start_date_str}"
    table_name += f"__end_{end_date_str}"
    table_name += "__" + "_".join(name_parts)

    full_name = sanitize_table_name(table_name)

    return full_name


def get_aggregated_table_name(
    segmentation: list[str],
    preauth_date_str: str = None,
) -> str:

    full_name = "ibnr_aggregated"

    if preauth_date_str:
        full_name = full_name + f"__preauth_{preauth_date_str}"

    if len(segmentation) == 0:
        segmentation_str = "__none"
    else:
        segmentation_str = "__" + "_".join(segmentation)

    full_name = full_name + segmentation_str

    full_name = sanitize_table_name(full_name)

    return full_name


def short_id(s, length=6):
    digest = hashlib.sha256(s.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode()[:length]


def get_model_name(
    prefix: str,
    table_name: str,
    target: str,
    feature_combination: list,
    model_type: str,
    start_date_str: str,
    end_date_str: str,
) -> tuple[str, str]:

    model_base_name = table_name.replace(f"{prefix}_gold", "ibnr")

    model_full_name = model_base_name + f"_type_{model_type}" + f"_target_{target}"
    model_full_name += f"__start_{start_date_str}"
    model_full_name += f"__end_{end_date_str}"
    all_features = "_".join(feature_combination)
    model_full_name += f"_features_{all_features}"

    model_name = f"{prefix}_{short_id(model_full_name)}"
    model_name = sanitize_dbx_name(model_name)

    assert len(model_name) < 20

    return model_name, model_full_name


def get_all_targets(run_type: str, relative_path: str = "../../../") -> list[str]:

    run_config = get_config_file("ibnr", "run_config", relative_path)

    all_targets = []
    model_types = run_config[run_type]
    for model_type, model_run_config in model_types["model_types"].items():
        all_targets += [x for x in model_run_config["targets"]]

    return list(set(all_targets))


def get_all_segmentation_combinations(run_type: str, relative_path: str = "../../../") -> list[str]:

    run_config = get_config_file("ibnr", "run_config", relative_path)

    all_combs = []
    model_types = run_config[run_type]
    for model_type, model_run_config in model_types["model_types"].items():
        all_combs += model_run_config["segmentation_combinations"]

    return all_combs


def get_all_run_types() -> list[str]:
    return ["ibnr_standalone", "ibnr_inflation"]


def get_run_config(
    relative_path: str = "../../../",
) -> dict:

    run_config = get_config_file("ibnr", "run_config", relative_path)

    per_lag_modeling = run_config["per_lag_modeling"]

    this_config = {}

    for run_type in get_all_run_types():
        model_types = run_config[run_type]
        for model_type, model_run_config in model_types["model_types"].items():
            for segmentation_combination in model_run_config["segmentation_combinations"]:
                segmentation_combination = tuple(segmentation_combination)
                if segmentation_combination not in this_config.keys():
                    this_config[segmentation_combination] = {}
                for training_years in model_run_config["nb_training_years"]:
                    if training_years not in this_config[segmentation_combination].keys():
                        this_config[segmentation_combination][training_years] = []
                    this_config[segmentation_combination][training_years].append(
                        {
                            "model_type": model_type,
                            "targets": model_run_config["targets"],
                            "input_features_combinations": model_run_config["input_features_combinations"],
                            "model_configs": model_run_config["model_configs"],
                            "ibnr_standalone": run_type == "ibnr_standalone",
                            "ibnr_inflation": run_type == "ibnr_inflation",
                        }
                    )

    all_timeframes = get_inference_timeframe_ranges(relative_path)

    aggregated_tables = {}
    gold_tables = {}
    models_created = []

    for this_timeframe in all_timeframes:
        this_start = this_timeframe["start_date"]
        this_end = this_timeframe["end_date"]
        this_start_str = this_timeframe["start_date_str"]
        this_end_str = this_timeframe["end_date_str"]
        this_preauth_date_str = this_timeframe["preauth_date_str"]
        this_max_lags = this_timeframe["max_lags"]

        for segmentation_combination in this_config.keys():
            aggregated_table_name = get_aggregated_table_name(list(segmentation_combination), this_preauth_date_str)
            aggregated_tables[aggregated_table_name] = {
                "table_name": aggregated_table_name,
                "segmentation_combination": list(segmentation_combination),
                "preauth_date_str": this_preauth_date_str,
                "max_lags": this_max_lags,
            }

            for training_years, inner_configs in this_config[segmentation_combination].items():
                table_name = get_gold_table_name(A
                    segmentation=list(segmentation_combination),
                    training_years=training_years,
                    start_date_str=this_start_str,
                    end_date_str=this_end_str,
                )
                gold_tables[table_name] = {
                    "table_name": table_name,
                    "segmentation_combination": list(segmentation_combination),
                    "aggregated_table_name": aggregated_table_name,
                    "training_years": training_years,
                    "start_date": this_start,
                    "end_date": this_end,
                }

                for inner_config in inner_configs:
                    for target in inner_config["targets"]:
                        for feature_combination in inner_config["input_features_combinations"]:
                            for model_config in inner_config["model_configs"]:
                                features = feature_combination.copy()
                                features = sorted(set(features))

                                model_type = inner_config["model_type"]
                                model_name, model_full_name = get_model_name(
                                    prefix="ibnr",
                                    table_name=table_name,
                                    target=target,
                                    feature_combination=features,
                                    model_type=model_type,
                                    start_date_str=this_start_str,
                                    end_date_str=this_end_str,
                                )

                                segmentation_columns = list(segmentation_combination).copy()
                                segmentation_with_possible_lag = segmentation_columns.copy()
                                is_per_lag_modeling = per_lag_modeling[model_type]
                                if is_per_lag_modeling:
                                    segmentation_with_possible_lag.append("lag")

                                job_compute = "job_cluster"

                                experiment_without_target = "PRED | "
                                experiment_without_target += " - ".join(features)
                                experiment_without_target += _get_model_experiment_name(model_type, model_config)
                                experiment_without_target += f"_{training_years}"
                                full_experiment = f"{experiment_without_target}_{target}"

                                models_created.append(
                                    {
                                        "table_name": table_name,
                                        "training_years": training_years,
                                        "target": target,
                                        "feature_combination": features,
                                        "model_name": model_name,
                                        "model_full_name": model_full_name,
                                        "model_config": str(model_config),
                                        "model_type": model_type,
                                        "is_per_lag_modeling": is_per_lag_modeling,
                                        "segmentation_columns_with_possible_lag": segmentation_with_possible_lag,
                                        "segmentation_columns": segmentation_columns,
                                        "ibnr_standalone": inner_config["ibnr_standalone"],
                                        "ibnr_inflation": inner_config["ibnr_inflation"],
                                        "job_compute": job_compute,
                                        "experiment_without_target": experiment_without_target,
                                        "full_experiment": full_experiment,
                                        "start_date": this_start,
                                        "end_date": this_end,
                                    }
                                )

    models_created = merge_duplicate_models(models_created)

    return aggregated_tables, gold_tables, models_created


def _get_model_experiment_name(model_type: str, model_config: dict) -> str:

    s = ""

    if model_type == "neural_network":
        s += " | NeuralNetworkConfig"
        s += " - "
        full_nn = model_config["unit_per_layer"].copy()
        full_nn.append(1)
        s += f"{full_nn}"
        s += " - "
        s += f"{model_config['activation']}"
        s += " - "
        s += f"{model_config['learning_rate']}"
        s += " - "
        s += f"{model_config['loss']}"
        s += " - "
        s += f"{model_config['epochs']}"
        s += " - "
        s += f"{model_config['batch_size']}"
    elif model_type == "chain_ladder":
        s += " | ChainLadderConfig"
    else:
        s += "model name not defined"

    return s


def merge_duplicate_models(models_created: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Group items by model_name (preserves order of first occurrence)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model in models_created:
        name = model["model_name"]
        groups[name].append(model)

    result: list[dict[str, Any]] = []

    for name, group in groups.items():
        if len(group) == 1:
            # Unique model → keep unchanged
            result.append(group[0])
            continue

        if len(group) != 2:
            # More than 2 duplicates for the same name is not allowed
            raise ValueError(f"Model name '{name}' appears {len(group)} times. Only exactly 1 or 2 occurrences are supported.")

        # Exactly 2 items → validate the required flag pattern and data equality
        standalone_item = None
        inflation_item = None

        for item in group:
            s = item["ibnr_standalone"]
            i = item["ibnr_inflation"]

            if s is True and i is False:
                if standalone_item is not None:
                    raise ValueError(f"Model name '{name}' has multiple items with ibnr_standalone=True / ibnr_inflation=False")
                standalone_item = item
            elif s is False and i is True:
                if inflation_item is not None:
                    raise ValueError(f"Model name '{name}' has multiple items with ibnr_standalone=False / ibnr_inflation=True")
                inflation_item = item
            else:
                raise ValueError(
                    f"Model name '{name}' contains an item with unexpected flag "
                    f"combination: ibnr_standalone={s}, ibnr_inflation={i}. "
                    "Only the pair (True/False) + (False/True) is allowed for duplicates."
                )

        if standalone_item is None or inflation_item is None:
            raise ValueError(f"Model name '{name}' does not contain exactly one item of each required flag pattern (standalone=True/inflation=False and the inverse).")

        # Verify all other keys have identical values
        keys_to_check = [k for k in standalone_item.keys() if k not in ("ibnr_standalone", "ibnr_inflation")]
        for key in keys_to_check:
            if standalone_item[key] != inflation_item[key]:
                raise ValueError(f"Model name '{name}' has differing values for key '{key}' between the two duplicate entries.")

        # All checks passed → merge
        # We copy the standalone item (shallow copy is sufficient and efficient
        # because the problem guarantees all other values are identical)
        merged = standalone_item.copy()
        merged["ibnr_standalone"] = True
        merged["ibnr_inflation"] = True
        result.append(merged)

    # Final original uniqueness check (exactly as you had before)
    all_names = [m["model_name"] for m in result]
    assert len(all_names) == len(set(all_names)), "Model names are not unique after duplicate merging. This should never happen with the logic above."

    return result


def get_inference_timeframe_ranges(relative_path: str = "../../../") -> list[dict[str, Any]]:

    config = get_config_file("ibnr", "run_config", relative_path)

    def _parse_one_timeframe(start: date, end: date) -> dict[str, Any]:
        max_lags = (end.year - start.year) * 12 + (end.month - start.month) + 1

        preauth_date = first_of_following_month(get_beginning_of_month_date(end))

        preauth_date_str = f"{preauth_date}".replace("-", "_")
        start_str = f"{start}".replace("-", "_")
        end_str = f"{end}".replace("-", "_")

        return {
            "start_date": start,
            "end_date": end,
            "start_date_str": start_str,
            "end_date_str": end_str,
            "preauth_date": preauth_date,
            "preauth_date_str": preauth_date_str,
            "max_lags": max_lags,
        }

    inference_timeframe = config["inference_timeframes"]

    use_custom_ranges = inference_timeframe["use_custom_ranges"]

    all_timeframes = []

    if use_custom_ranges:
        for one_range in inference_timeframe["ranges"]:
            begin_range = _parse_str_to_date(one_range["begin"])
            end_range = _parse_str_to_date(one_range["end"])

            all_timeframes.append(_parse_one_timeframe(begin_range, end_range))
    else:
        input_date = datetime.today()

        begin_range = (input_date - relativedelta(years=1)).replace(day=1)
        end_range = (input_date - relativedelta(months=1)).replace(day=1)

        all_timeframes.append(_parse_one_timeframe(begin_range.date(), end_range.date()))

    all_max_lags = [x["max_lags"] for x in all_timeframes]
    if 1 != len(list(set(all_max_lags))):
        raise ValueError(f"All time ranges must have the same number of months.")

    return all_timeframes


def get_maximal_years_runoff_lags() -> int:

    config = get_config_file("ibnr", "feature_engineering_config")

    return config["maximal_years_runoff_lags"]


def get_min_available_timeframe() -> str:

    config = get_config_file("ibnr", "feature_engineering_config")

    return config["min_available_timeframe"]


def get_occured_date_col(relative_path: str = "../../../") -> str:

    config = get_config_file("ibnr", "feature_engineering_config", relative_path)

    return config["occured_date_col"]
