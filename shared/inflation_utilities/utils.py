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
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from collections import defaultdict
from shared_utilities.utils import (
    get_config_file,
    _parse_str_to_bool,
    _parse_str_to_date,
)

def sanitize_dbx_name(name: str) -> str:

    sanitized = re.sub(r'[.\s/()[\]{}|]|[\x00-\x1F\x7F]', '_', name)
    sanitized = sanitized.replace('-', '_')

    # Force lowercase (Unity Catalog stores names in lowercase)
    return sanitized.lower()

def sanitize_table_name(name: str) -> str:
    if len(name) > 255:
        raise ValueError('Databricks/UC only allow table names up to 255 chars')

    return sanitize_dbx_name(name)

def get_is_to_compensate_ibnr_on_invoices_view() -> bool:

    config = get_config_file('inflation', 'run_config')

    return _parse_str_to_bool(config['is_to_compensate_ibnr_on_invoices_view'])

def get_timeframes(relative_path: str = '../../../') -> dict[str, Any]:

    config = get_config_file('inflation', 'run_config', relative_path)

    timeframe = config['timeframe_info']

    use_custom_ranges = timeframe['use_custom_ranges']

    if use_custom_ranges:
        nb_forecast_years = timeframe['nb_forecast_years']
        training_begin_range = _parse_str_to_date(timeframe['training_begin'])
        training_end_range = _parse_str_to_date(timeframe['training_end'])
        inference_begin_range = (training_end_range + relativedelta(months=1)).replace(day=1)
        inference_end_range = (inference_begin_range + relativedelta(years=nb_forecast_years) - relativedelta(months=1)).replace(day=1)
    else:
        input_date = datetime.today()
        nb_forecast_years = 3
        historical_months = 10 * 12 + 3

        inference_begin_range = (input_date).replace(day=1).date()
        inference_end_range = (input_date + relativedelta(years=nb_forecast_years) - relativedelta(months=1)).replace(day=1).date()
        training_begin_range = (input_date - relativedelta(months=historical_months)).replace(day=1).date()
        training_end_range = (input_date - relativedelta(months=1)).replace(day=1).date()

    all_timeframes = {
        'inference_begin_range': inference_begin_range,
        'inference_end_range': inference_end_range,
        'training_begin_range': training_begin_range,
        'training_end_range': training_end_range,
    }

    return all_timeframes

def get_run_config(spark) -> dict:
    
    df = (
        spark
        .table('axahealth_dataplatform_pr_gold.data_science.actuarial_ml_risk_clusters')
    )

    cluster_id_column = 'cluster_id'
    model_config_column = 'model_config'
    target_column = 'target'

    df = TEMPORARY_FUNCTION_ADD_COLUMNS(spark, df, cluster_id_column, model_config_column, target_column)

    filtering_columns = [c for c in df.columns if c not in[cluster_id_column, model_config_column, target_column]]

    rows = df.collect()

    all_cluster_ids = [row["cluster_id"] for row in rows]
    assert len(all_cluster_ids) == len(set(all_cluster_ids)), "cluster_id is not unique across the dataframe!"
    assert len(rows) == len(set(all_cluster_ids)), "cluster_id is not unique across the dataframe!"

    cluster_tables = {}
    gold_tables = {}
    models_created = {}

    for row in rows:
        this_target = row[target_column]
        this_cluster_id = row[cluster_id_column]

        this_model_config = {}
        for k, v in row[model_config_column].items():
            if v is not None:
                this_model_config[k] = json.loads(v)
            else:
                this_model_config[k] = v
        use_regressors = False
        if 'regressors' in this_model_config and this_model_config['regressors'] is not None:
            use_regressors = True

        this_filtering_values = {col: list(row[col]) for col in filtering_columns}

        cluster_table_name = f'cluster_{this_cluster_id}'
        cluster_table_name = sanitize_table_name(cluster_table_name)
        cluster_tables[cluster_table_name] = {
            cluster_id_column: this_cluster_id,
            'table_name': cluster_table_name,
            'use_regressors': use_regressors,
            'filtering_values': this_filtering_values,
        }

        gold_table_name = f'inflation_gold_{this_target}'
        gold_table_name = sanitize_table_name(gold_table_name)
        if gold_table_name not in gold_tables:
            gold_tables[gold_table_name] = {'table_name': gold_table_name, 'clusters': []}
        gold_tables[gold_table_name]['clusters'].append({
            cluster_id_column: this_cluster_id,
            'filtering_values': this_filtering_values,
            'cluster_table_name': cluster_table_name,
        })

        model_name = f'inflation_{this_target}'
        model_name = sanitize_dbx_name(model_name)
        if model_name not in models_created:
            models_created[model_name] = {'model_name': model_name, 'target': this_target, 'table_name': gold_table_name, 'clusters': []}
        models_created[model_name]['clusters'].append({
            cluster_id_column: this_cluster_id,
            'filtering_values': this_filtering_values,
            model_config_column: this_model_config,
        })

    assert(len(cluster_tables) == len(rows))
    total_internal_segmentations = 0
    for model_info in models_created.values():
        total_internal_segmentations += len(model_info['clusters'])
    assert(total_internal_segmentations == len(rows))
    assert(len(gold_tables) == len(models_created))

    run = {}
    run['cluster_tables'] = cluster_tables
    run['gold_tables'] = gold_tables
    run['models_created'] = models_created

    return run

def TEMPORARY_FUNCTION_ADD_COLUMNS(spark, df: DataFrame, cluster_id: str, model_config: str, target) -> DataFrame:

    df = (
        df.withColumn(target, F.lit('Frequency_claims'))
        .unionByName(df.withColumn(target, F.lit('BC')))
    )

    w_id = W.orderBy(cluster_id, target)
    filter_columns = [c for c in df.columns if c not in[cluster_id, model_config, target]]
    simple_config_one = {
        'regressors': {'price_CPI': 10},
        'changepoint_nb_months_censored': 1,
        'changepoint_prior_scale': 1.5,
    }
    simple_config_two = {
        'regressors': None,
        'changepoint_nb_months_censored': 0,
        'changepoint_prior_scale': 5.5,
    }
    df = (
        df
        .withColumn(
            cluster_id,
            F.concat_ws('__', *filter_columns)
        )
        .withColumn(
            cluster_id,
            F.row_number().over(w_id)
        )
        .withColumn(
            model_config,
            F.when(
                F.col(cluster_id) % 3 == 0,
                F.lit(json.dumps(simple_config_one))
            )
            .otherwise(F.lit(json.dumps(simple_config_two)))
        )
        .withColumn(
            model_config,
            F.from_json(F.col(model_config), MapType(StringType(), StringType()))
        )
        .withColumn(
            cluster_id,
            F.concat_ws('_', F.lit('cluster'), F.col(cluster_id))
        )
    )

    return df







