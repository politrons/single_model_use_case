import json
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import yaml
from typing import Any

def get_config_file(use_case: str, config_file: str, relative_path: str = '../../../') -> dict:
    with open(f'{relative_path}business_case_config/{use_case}/{config_file}.yml', 'r', encoding='utf-8') as yaml_config:
        config = yaml.safe_load(yaml_config)
    return config

def get_maximal_years_runoff_lags() -> int:

    config = get_config_file('ibnr', 'feature_engineering_config')

    return config['maximal_years_runoff_lags']

def get_full_table_name(table_alias: str, use_case: str = 'ibnr') -> str:

    config = get_config_file(use_case, 'data_config')

    return config['tables'][table_alias]['full_name']

def get_primary_keys(table_alias: str, use_case: str = 'ibnr') -> list[str]:

    config = get_config_file(use_case, 'data_config')

    return config['tables'][table_alias]['primary_keys']

def get_min_available_timeframe() -> str:

    config = get_config_file('ibnr', 'feature_engineering_config')

    return config['min_available_timeframe']

def _parse_str_to_date(this_date: str | date | datetime) -> date:
    match this_date:
        case datetime() as this_datetime:
            return this_datetime.date()
        case date() as this_date_actual:
            return this_date_actual
        case str() as this_date_str:
            return datetime.strptime(this_date_str.replace('_', '-'), "%Y-%m-%d").date()
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

    true_values = ['True', 'true', True, 1]
    false_values = ['False', 'false', False, 0]
    
    if this_bool in true_values:
        return True
    elif this_bool in false_values:
        return False
    else:
        raise TypeError("format not recognised")

def get_inference_timeframe_ranges(use_case: str, relative_path: str = '../../../') -> list[dict[str, Any]]:

    ## TODO: Move this to ibnr utlities and remove the use_case parameter

    config = get_config_file(use_case, 'run_config', relative_path)

    def _parse_one_timeframe(start: date, end: date) -> dict[str, Any]:
        max_lags = (end.year - start.year) * 12 + (end.month - start.month) + 1

        preauth_date = first_of_following_month(get_beginning_of_month_date(end))

        preauth_date_str = f'{preauth_date}'.replace('-', '_')
        start_str = f'{start}'.replace('-', '_')
        end_str = f'{end}'.replace('-', '_')

        return {
            'start_date': start,
            'end_date': end,
            'start_date_str': start_str,
            'end_date_str': end_str,
            'preauth_date': preauth_date,
            'preauth_date_str': preauth_date_str,
            'max_lags': max_lags,
        }

    inference_timeframe = config['inference_timeframes']

    use_custom_ranges = inference_timeframe['use_custom_ranges']

    all_timeframes = []

    if use_custom_ranges:
        for one_range in inference_timeframe['ranges']:
            begin_range = _parse_str_to_date(one_range['begin'])
            end_range = _parse_str_to_date(one_range['end'])

            all_timeframes.append(_parse_one_timeframe(begin_range, end_range))
    else:
        input_date = datetime.today()

        if use_case == 'ibnr':
            begin_range = (input_date - relativedelta(years=1)).replace(day=1)
            end_range = (input_date - relativedelta(months=1)).replace(day=1)
        else:
            raise ValueError('unrecognized use case')

        all_timeframes.append(_parse_one_timeframe(begin_range.date(), end_range.date()))

    all_max_lags = [x['max_lags'] for x in all_timeframes]
    if 1 != len(list(set(all_max_lags))):
        raise ValueError(f"All time ranges must have the same number of months.")

    return all_timeframes

def get_occured_date_col(relative_path: str = '../../../') -> str:

    config = get_config_file('ibnr', 'feature_engineering_config', relative_path)

    return config['occured_date_col']

def get_temporal_column(relative_path: str = '../../../') -> str:

    config = get_config_file('inflation', 'feature_engineering_config', relative_path)

    return config['temporal_column']




