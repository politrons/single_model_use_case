# Training Module

## Overview

This module contains multiple training notebooks that illustrate different **training strategies** (single model, multi‑model, Optuna hyper‑parameter tuning, Feature Store integration, Medallion architecture, etc.).

## Package map

- `train_model.py`
  - Main training entrypoint used by orchestrated jobs.
- `data/`
  - `training_data_config.py`: config-driven training data loader.
  - `training_data_contract.py`: custom training data loader contract.
- `model/`
  - `model_config.py`: config-driven model builder.
  - `model_contract.py`: custom model contract.
  - `predict_and_proba_wrapper.py`: wrapper exposing `prediction` and `prediction_proba`.
  - `algorithms/`: algorithm factory/adapters (including TensorFlow adapter).
  - `hyperparam_searches/`: search backends (`scikit`, `ray`) and base API.
  - `pipeline/`: pipeline builders (including early stop pipeline).
  - `samplers/`: imbalance sampling adapters.
  - `transformers/`: feature/target transformation components.


## Expected Input Parameters

The notebooks rely on the following *Databricks notebook widgets* (or equivalent job parameters).

| Parameter                | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `catalog`                | Unity Catalog catalog where tables are stored.               |
| `data_path`              | Path to a Delta table or external data source.               |
| `dropoff_features_table` | Unity Catalog table name.                                    |
| `env`                    | Target environment (dev / staging / prod).                   |
| `experiment`             | Name or path of the MLflow experiment.                       |
| `experiment_name`        | Name or path of the MLflow experiment.                       |
| `metrics_latency_table`  | Unity Catalog table name.                                    |
| `model_algorithm_path`   | Path or identifier.                                          |
| `model_card_path`        | Path or identifier.                                          |
| `model_name`             | Name of the registered MLflow model.                         |
| `pickup_features_table`  | Unity Catalog table name.                                    |
| `random_state`           | Seed used for deterministic splitting and random operations. |
| `schema`                 | Database/schema name.                                        |
| `test_size`              | Fraction of the dataset reserved for testing.                |
| `training_data_path`     | Path to a Delta table or external data source.               |

## Implementation Contracts

The module defines the following **abstract base classes** that serve as contracts. Users **must implement** all abstract methods when extending the stack:

### `ModelContract` (training/model\_algorithms/model\_contract.py)

| Method                | Parameters                                    | Description                    |
| --------------------- | --------------------------------------------- | ------------------------------ |
| `get_model_algorithm` |                                               | Custom implementation required |
| `log_model`           | `model, model_name, signature, input_example` | Custom implementation required |

### `MultiModelContract` (training/model\_algorithms/multi\_model\_contract.py)

| Method         | Parameters                                    | Description                    |
| -------------- | --------------------------------------------- | ------------------------------ |
| `get_models`   |                                               | Custom implementation required |
| `select_model` | `models_evaluations`                          | Custom implementation required |
| `log_model`    | `model, model_name, signature, input_example` | Custom implementation required |

### `TrainingDataContract` (training/data/training\_data\_contract.py)

| Method                    | Parameters | Description                    |
| ------------------------- | ---------- | ------------------------------ |
| `get_training_data`       | `raw_data` | Custom implementation required |
| `transform_training_data` | `x, y`     | Custom implementation required |

---

## Predict + Predict Proba Support

The training pipeline now supports logging both class predictions and probabilities.

### 1. Optional model config flag

In `model_config`, you can set:

```yaml
prediction_method: predict_proba
```

If this flag is missing, the pipeline keeps the previous behavior and uses the model as-is.

### 2. Wrapper mode (`prediction_method: predict_proba`)

When `prediction_method` is set to `predict_proba`:

* The model is wrapped by `PredictAndProbaWrapper` (`training/model/predict_and_proba_wrapper.py`).
* The wrapped model keeps your public model name and stores the original model under `internal_<model_name>`.

In this mode, `predict()` returns a DataFrame with:

* `prediction`
* `prediction_proba`

If the model does not implement `predict_proba`, inference fails at runtime when the wrapper calls `predict_proba`.

### 3. User-defined model outputs (without wrapper mode)

Even without `prediction_method: predict_proba`, the framework can detect probabilities if
`model.predict(...)` returns structured outputs such as:

* DataFrame (including two-column prediction + probability shapes)
* `dict` with keys like `prediction`/`predictions` and `prediction_proba`/`probability`/`proba`
* tuple `(prediction, probability)`

If probabilities are detected, they are propagated through signature and baseline generation.

### 4. Baseline table behavior

Baseline generation keeps `prediction` behavior unchanged and adds `prediction_proba` only when available.

* Column name: `prediction_proba`
* Values are normalized/serialized so vector probabilities can be stored safely.
* If probabilities are not present, the column is not added to baseline rows.

### 5. Signature behavior

Model signature inference includes:

* `prediction` always
* `prediction_proba` only when probabilities are present in model output
