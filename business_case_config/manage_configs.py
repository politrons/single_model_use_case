# Databricks notebook source
# DBTITLE 1,Add path
import sys, os
_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../shared"))
sys.path.insert(0, _shared)


# COMMAND ----------

# DBTITLE 1,Imports
from pathlib import Path
import shutil
import json
import yaml
import importlib.util
import itertools
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from pyspark.sql import functions as F
from functools import reduce
import sys
import yaml
from inflation_utilities.utils import (
    get_temporal_column,
)
from ibnr_utilities.utils import (
    get_occured_date_col,
)
from ibnr_utilities.utils import (
    get_run_config as get_ibnr_run_config,
)
from inflation_utilities.utils import (
    get_run_config as get_inflation_run_config,
)
from contract_utilities.generate_contract import (
    generate_ibnr_contract,
    generate_inflation_contract,
)


# COMMAND ----------

configs_path = Path("../configs/")
contracts_path = Path("../contracts/")

module_path = Path("../feature_store/01_ibnr_pipe/utilities/feature_engineering.py")
spec = importlib.util.spec_from_file_location("your_module", module_path)
custom_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_module)

random_state = 42
source_schema = 'health_data_factory_forecast_framework'

# COMMAND ----------

models_to_be_deleted = [f.name for f in configs_path.iterdir() if f.is_dir()]

if models_to_be_deleted:
    print(f"Deleting {len(models_to_be_deleted)} model configs...\n")

    for i, model_name in enumerate(models_to_be_deleted):
        model_path = configs_path / model_name
        print(f"{i}: Deleting model {model_name} ...")
        shutil.rmtree(model_path, ignore_errors=True)   # safely delete folder + contents

    print("All old model configs deleted.\n")
else:
    print("No existing models found to delete.\n")

models_to_be_deleted = [f.name for f in contracts_path.iterdir() if f.is_dir()]

if models_to_be_deleted:
    print(f"Deleting {len(models_to_be_deleted)} model contracts...\n")

    for i, model_name in enumerate(models_to_be_deleted):
        model_path = contracts_path / model_name
        print(f"{i}: Deleting model {model_name} ...")
        shutil.rmtree(model_path, ignore_errors=True)   # safely delete folder + contents

    print("All old model contracts deleted.\n")
else:
    print("No existing models found to delete.\n")

# COMMAND ----------

# DBTITLE 1,Get IBNR models info
ibnr_info = {}
(
    ibnr_info['aggregated_tables'],
    ibnr_info['gold_tables'],
    ibnr_info['models_created'], 
) = get_ibnr_run_config(
    relative_path='../',
)

# COMMAND ----------

# DBTITLE 1,Create IBNR configs and contracts
print(f"Creating {len(ibnr_info['models_created'])} ...\n")

temporal_reference_column = get_occured_date_col(relative_path='../')

possible_later_addition_to_aux = [
    "cumulative_amount_paid",
    "cumulative_nb_claim_paid",
    "cumulative_nb_invoice_paid",
    'lag',
]
ibnr_models_created = []
for model_info in ibnr_info['models_created']:
    model_name = model_info['model_name']
    model_type = model_info['model_type']
    target = model_info['target']
    model_table_source = model_info['table_name']
    training_years = model_info['training_years']
    segmentation_columns = model_info['segmentation_columns_with_possible_lag']
    model_config = model_info['model_config']
    if model_config is not None and model_config != 'None':
        model_config = json.loads(model_config.replace("'", '"'))
    print(f"Creating model {model_name} ...")
    model_path = configs_path / model_name
    model_path.mkdir(exist_ok=True)
    model_contract_path = contracts_path / model_name
    model_contract_path.mkdir(exist_ok=True)
    features = model_info['feature_combination'].copy()
    chain_ladder_required_columns = [temporal_reference_column, 'lag'] if model_type == 'chain_ladder' else []
    auxiliary_columns = [temporal_reference_column, 'segment', 'segment_key', 'is_inference']
    to_be_added_aux = [
        x
        for x in possible_later_addition_to_aux
        if x not in list(features + auxiliary_columns + segmentation_columns)
    ]
    job_config = {
        'custom_jobs': {
            'op-job': {
                'trigger_stages': [
                    'factory-job',
                    'batch-job',
                ],
                'environments': ['staging', 'prod'],
            },
        }
    }
    with open(model_path / "job_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(job_config, f, sort_keys=False, default_flow_style=False, indent=2)
    final_feature_columns = sorted(set(features + segmentation_columns + chain_ladder_required_columns))
    final_auxiliary_columns = [
        col
        for col in sorted(set(auxiliary_columns + to_be_added_aux))
        if col not in final_feature_columns
    ]
    training_data_config = {
        'schema': source_schema,
        'table_name': model_table_source,
        'feature_columns': final_feature_columns,
        'target_column': target,
        'auxiliary_columns': final_auxiliary_columns,
        'temporal_reference_column': temporal_reference_column,
    }
    with open(model_path / "training_data_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(training_data_config, f, sort_keys=False, default_flow_style=False, indent=2)
    split_config = {
        'split_strategy': 'time_series',
        'temporal_reference_column': temporal_reference_column,
        'train_size': (training_years * 12) / ((training_years + 1) * 12)
    }
    with open(model_path / "split_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(split_config, f, sort_keys=False, default_flow_style=False, indent=2)
    contract_numerical_features = (
        sorted(set(features + chain_ladder_required_columns))
        if model_type == 'chain_ladder'
        else features
    )
    generate_ibnr_contract(
        model_contract_path=model_contract_path,
        model_type=model_type,
        numerical_features=contract_numerical_features,
        segment_columns=segmentation_columns,
        model_config=model_config,
        temporal_reference_column=temporal_reference_column,
        random_state=random_state,
        n_jobs=-1,
        relative_path='../',
    )

    validation_config = {
        'model_type': "regressor",
        'targets': target,
        'target_type': "float",
    }
    with open(model_path / "validation_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(validation_config, f, sort_keys=False, default_flow_style=False, indent=2)
    eval_result_config = {
        "r2_score": {
            "op": "<=",
            "value": 1.0,
        },
    }
    with open(model_path / "eval_result_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(eval_result_config, f, sort_keys=False, default_flow_style=False, indent=2)
    integration_test_config = {
        'columns': features,
        'data': [0] * len(features),
        'prediction': {
            'op': '>=',
            'value': 0.0,
        }
    }
    with open(model_path / "integration_test_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(integration_test_config, f, sort_keys=False, default_flow_style=False, indent=2)
    quality_monitor_config = {
        'baseline_enable': False,
        'schema_name': source_schema,
        'table_name': model_table_source,
        'granularities': "1 day,1 week",
        'match_columns': [temporal_reference_column, 'lag', 'segment_key'],
        'target_col': target,
    }
    with open(model_path / "quality_monitor_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(quality_monitor_config, f, sort_keys=False, default_flow_style=False, indent=2)
    batch_connector_config = {
        'batch_connector_enable': True,
        'batch_pagination_size': 500,
        'filter_by_target': False,
    }
    with open(model_path / "batch_connector_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(batch_connector_config, f, sort_keys=False, default_flow_style=False, indent=2)
    with open(model_path / "model_tuning_config.yml", "w", encoding="utf-8") as f:
        yaml.dump({}, f, sort_keys=False, default_flow_style=False, indent=2)
    with open(model_path / "performance_test_config.yml", "w", encoding="utf-8") as f:
        yaml.dump({}, f, sort_keys=False, default_flow_style=False, indent=2)
    ibnr_models_created.append(model_name)
assert(len([f.name for f in configs_path.iterdir() if f.is_dir()]) == len(ibnr_models_created))
print(f"\n Models successfully recreated: {len(ibnr_models_created)}")


# COMMAND ----------

# DBTITLE 1,Get Inflation models info
inflation_info = get_inflation_run_config(spark, relative_path='../')


# COMMAND ----------

print(f"Creating {len(inflation_info['models_created'])} ...\n")
temporal_reference_column = get_temporal_column(relative_path='../')
segment_column = 'cluster_id'
inflation_models_created = []
for model_info in inflation_info['models_created'].values():
    model_name = model_info['model_name']
    target = model_info['target']
    model_table_source = model_info['table_name']
    clusters = model_info.get('clusters', [])
    print(f"Creating model {model_name} ...")
    model_path = configs_path / model_name
    model_path.mkdir(exist_ok=True)
    model_contract_path = contracts_path / model_name
    model_contract_path.mkdir(exist_ok=True)
    inflation_contract_info = generate_inflation_contract(
        model_contract_path=model_contract_path,
        clusters=clusters,
        temporal_reference_column=temporal_reference_column,
        segment_column=segment_column,
        random_state=random_state,
        n_jobs=-1,
        relative_path='../',
    )
    feature_columns = inflation_contract_info['feature_columns']
    regressor_columns = inflation_contract_info['regressor_columns']
    job_config = {
        'custom_jobs': {
            'op-job': {
                'trigger_stages': [
                    'factory-job',
                    'batch-job',
                ],
                'environments': ['staging', 'prod'],
            },
        }
    }
    with open(model_path / "job_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(job_config, f, sort_keys=False, default_flow_style=False, indent=2)
    training_data_config = {
        'schema': source_schema,
        'table_name': model_table_source,
        'feature_columns': feature_columns,
        'target_column': target,
        'auxiliary_columns': [],
        'temporal_reference_column': temporal_reference_column,
    }
    with open(model_path / "training_data_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(training_data_config, f, sort_keys=False, default_flow_style=False, indent=2)

    split_config = {
        'split_strategy': 'time_series',
        'temporal_reference_column': temporal_reference_column,
        'train_size': 0.8,
    }
    with open(model_path / "split_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(split_config, f, sort_keys=False, default_flow_style=False, indent=2)
    validation_config = {
        'model_type': 'regressor',
        'targets': target,
        'target_type': 'float',
    }
    with open(model_path / "validation_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(validation_config, f, sort_keys=False, default_flow_style=False, indent=2)
    eval_result_config = {
        'r2_score': {
            'op': '<=',
            'value': 1.0,
        },
    }
    with open(model_path / "eval_result_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(eval_result_config, f, sort_keys=False, default_flow_style=False, indent=2)
    integration_columns = [temporal_reference_column] + list(regressor_columns)
    integration_data = [
        '2024-01-01' if col == temporal_reference_column else 0.0
        for col in integration_columns
    ]
    integration_test_config = {
        'columns': integration_columns,
        'data': integration_data,
        'prediction': {
            'op': '>=',
            'value': 0.0,
        }
    }
    with open(model_path / "integration_test_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(integration_test_config, f, sort_keys=False, default_flow_style=False, indent=2)
    quality_monitor_config = {
        'baseline_enable': False,
        'schema_name': source_schema,
        'table_name': model_table_source,
        'granularities': '1 day,1 week',
        'match_columns': [temporal_reference_column, segment_column],
        'target_col': target,
    }
    with open(model_path / "quality_monitor_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(quality_monitor_config, f, sort_keys=False, default_flow_style=False, indent=2)
    batch_connector_config = {
        'batch_connector_enable': True,
        'batch_pagination_size': 500,
        'filter_by_target': False,
    }
    with open(model_path / "batch_connector_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(batch_connector_config, f, sort_keys=False, default_flow_style=False, indent=2)
    with open(model_path / "model_tuning_config.yml", "w", encoding="utf-8") as f:
        yaml.dump({}, f, sort_keys=False, default_flow_style=False, indent=2)
    with open(model_path / "performance_test_config.yml", "w", encoding="utf-8") as f:
        yaml.dump({}, f, sort_keys=False, default_flow_style=False, indent=2)

    inflation_models_created.append(model_name)
print(f"\n Models successfully recreated: {len(inflation_models_created)}")


# COMMAND ----------

feature_store_path = Path("../feature_store/")

ibnr_models_name = [x['model_name'] for x in ibnr_info['models_created'] if x['model_name'] in ibnr_models_created]
ibnr_op_jobs = [f'{x}-op-job' for x in ibnr_models_name]
ibnr_monitor_jobs = [f'{x}-monitor-job' for x in ibnr_models_name]

shared_pipe_name = 'shared_pipe'
shared_pipe_folder = '00_' + shared_pipe_name
shared_pipe_location = feature_store_path / shared_pipe_folder
shared_pipe_config = {
    'name': shared_pipe_name,
    'script_path': "transformations/",
    'cron_job': "0 0 1 1 * ?",
    'batch_run': True,
}
with open(shared_pipe_location / "pipeline_config.yml", "w", encoding="utf-8") as f:
    yaml.dump(shared_pipe_config, f, sort_keys=False, default_flow_style=False, indent=2)

ibnr_pipe_name = 'ibnr_pipe'
ibnr_pipe_folder = '01_' + ibnr_pipe_name
ibnr_pipe_location = feature_store_path / ibnr_pipe_folder
ibnr_pipe_config = {
    'name': ibnr_pipe_name,
    'script_path': "transformations/",
    'cron_job': "0 0 2 1 * ?",
    'batch_run': True,
}
with open(ibnr_pipe_location / "pipeline_config.yml", "w", encoding="utf-8") as f:
    yaml.dump(ibnr_pipe_config, f, sort_keys=False, default_flow_style=False, indent=2)

ibnr_out_name = 'ibnr_output'
ibnr_out_folder = '02_' + ibnr_out_name
ibnr_out_location = feature_store_path / ibnr_out_folder
ibnr_out_config = {
    'name': ibnr_out_name,
    'batch_run': True,
    'script_path': "transformations/",
    # 'cron_job': "0 0 3 1 * ?",
    'environments': [
        'staging',
        'prod'
    ],
    'custom_jobs': {
        'run-job': {
            'trigger_stages': [
                shared_pipe_name,
                ibnr_pipe_name,
                ibnr_op_jobs,
            ],
            'cron_job': "0 0 3 2 * ?",
            ########## maybe set also for the last day previous month, to make sure everything is correct?
            ########## that way you get a 'warning'
            'environments': ['staging', 'prod'],
        },
        'out-job': {
            'trigger_stages': [ibnr_monitor_jobs, ibnr_out_name],
            'cron_job': "0 0/30 4-10 2 * ?",
            'environments': ['staging', 'prod'],
        },
    },
}
with open(ibnr_out_location / "pipeline_config.yml", "w", encoding="utf-8") as f:
    yaml.dump(ibnr_out_config, f, sort_keys=False, default_flow_style=False, indent=2)

inflation_pipe_name = 'inflation_pipe'
inflation_pipe_folder = '03_' + inflation_pipe_name
inflation_pipe_location = feature_store_path / inflation_pipe_folder
inflation_pipe_config = {
    'name': inflation_pipe_name,
    'script_path': "transformations/",
    # 'cron_job': "0 0 4 1 * ?",
}
with open(inflation_pipe_location / "pipeline_config.yml", "w", encoding="utf-8") as f:
    yaml.dump(inflation_pipe_config, f, sort_keys=False, default_flow_style=False, indent=2)

# COMMAND ----------

models_orchestrator = {}

with open("../models_orchestrator.yml", "w", encoding="utf-8") as f:
    yaml.dump(models_orchestrator, f, sort_keys=False, default_flow_style=False, indent=2)
