# Agent Guide: How `model_contract_impl.py` Is Generated

## Purpose
This document explains the full flow that generates `model_contract_impl.py`, starting from:
- `business_case_config/*/run_config.yml`
- `business_case_config/manage_configs.py`
- helper modules under `shared/`

It also covers the `chain_ladder` (`ibnr_cl`) special case and why `KeyError: 'treatment_date'` can happen.

## High-level flow
1. Model definitions are declared in `run_config.yml`.
2. Combinations are expanded (targets, features, segmentations, time windows) in `shared/ibnr_utilities/utils.py` and `shared/inflation_utilities/utils.py`.
3. Config and contract files are orchestrated/written by `business_case_config/manage_configs.py`.
4. Final contract composition happens in `shared/contract_utilities/generate_contract.py` using templates from `contract_templates/`.
5. Output is written to `contracts/<model_name>/training/model/model_contract_impl.py`.

## 1) Input: `run_config.yml`
### IBNR
File: `business_case_config/ibnr/run_config.yml`

Per model type (`neural_network`, `chain_ladder`), it defines:
- `model_configs`
- `segmentation_combinations`
- `targets`
- `nb_training_years`
- `input_features_combinations`

### Inflation
File: `business_case_config/inflation/run_config.yml`

Same concept, with cluster/target structure for the inflation use case.

## 2) Model expansion in `shared/`
### IBNR
File: `shared/ibnr_utilities/utils.py`, function `get_run_config(...)`.

This function:
- reads `run_config.yml`
- computes the cross-product of segmentation/target/features/model_config/training_years
- applies time windows
- builds `models_created` with per-model metadata (`model_type`, `feature_combination`, `segmentation_columns_with_possible_lag`, etc.)

Outputs:
- `aggregated_tables`
- `gold_tables`
- `models_created`

### Inflation
File: `shared/inflation_utilities/utils.py`, function `get_run_config(...)`.

Returns model metadata with cluster-level settings used to build Prophet contracts.

## 3) Orchestration in `manage_configs.py`
File: `business_case_config/manage_configs.py`

This script:
- removes existing model folders under `configs/*` and `contracts/*`
- loads `ibnr_info` and `inflation_info`
- writes per-model config files (`training_data_config.yml`, `split_config.yml`, `validation_config.yml`, etc.)
- calls contract generators:
  - `generate_ibnr_contract(...)`
  - `generate_inflation_contract(...)`

### Important IBNR details
For each `model_info`:
- `features = model_info['feature_combination']`
- `segmentation_columns = model_info['segmentation_columns_with_possible_lag']`
- it builds:
  - `final_feature_columns`
  - `final_auxiliary_columns`
- then calls `generate_ibnr_contract(...)` with `numerical_features`.

## 4) Contract generation in `generate_contract.py`
File: `shared/contract_utilities/generate_contract.py`

### 4.1 `generate_ibnr_contract(...)` / `generate_inflation_contract(...)`
These functions prepare:
- `numerical_features`
- `segment_columns`
- `config`
- `extra_params`

Then they delegate to `generate_model_contract(...)`.

### 4.2 `generate_model_contract(...)`
Builds one self-contained `model_contract_impl.py` by inlining:
1. `contract_templates/multi_cluster_wrapper.py`
2. model-specific implementation from `_MODEL_TYPE_TO_PATH`:
   - `tensorflow` -> `contract_templates/ibnr_nn.py`
   - `chain_ladder` -> `contract_templates/ibnr_cl.py`
   - `prophet` -> `contract_templates/inflation_prophet.py`
3. framework template `contract_templates/framework.py`

It also:
- merges imports (deduplicated)
- injects placeholders (`{{NUMERICAL_FEATURES}}`, `{{CONFIG}}`, etc.)
- writes output to:
  - `contracts/<model_name>/training/model/model_contract_impl.py`

## 5) `chain_ladder` (`ibnr_cl`) special case
Internal template: `contract_templates/ibnr_cl.py`

In `fit(...)`, sorting/grouping uses:
- segmentation columns
- `occurrence_date_col`
- `lag_col`

If `X` does not contain `occurrence_date_col` or `lag_col`, `KeyError` is raised.

### How those fields must reach the contract
Current expected flow:
- `run_config.yml` defines `occurrence_date_col` and optionally `lag_col` in `chain_ladder.model_configs`
- `manage_configs.py` ensures chain ladder required columns are included in `feature_columns` / `numerical_features`
- `generate_ibnr_contract(...)` ensures both `numerical_features` and `extra_params.feature_columns` include:
  - `occurrence_date_col`
  - `lag_col`

## 6) Why `KeyError: 'treatment_date'` happens
Usually one of these:
- `treatment_date` never made it into `feature_columns`
- it was dropped during column filtering before model fit
- chain ladder contract was generated without `treatment_date` in `_NUMERICAL_FEATURES`

Quick checklist:
1. Verify `business_case_config/ibnr/run_config.yml` under `chain_ladder.model_configs`:
   - `occurrence_date_col: treatment_date`
   - `lag_col: lag`
2. Run `business_case_config/manage_configs.py` to regenerate configs/contracts.
3. Open generated contract and confirm `_NUMERICAL_FEATURES` includes `treatment_date` and `lag`.
4. Redeploy the bundle/job.

## 7) Key files
- `business_case_config/ibnr/run_config.yml`
- `business_case_config/inflation/run_config.yml`
- `business_case_config/manage_configs.py`
- `shared/ibnr_utilities/utils.py`
- `shared/inflation_utilities/utils.py`
- `shared/contract_utilities/generate_contract.py`
- `contract_templates/multi_cluster_wrapper.py`
- `contract_templates/ibnr_nn.py`
- `contract_templates/ibnr_cl.py`
- `contract_templates/inflation_prophet.py`
- `contract_templates/framework.py`
