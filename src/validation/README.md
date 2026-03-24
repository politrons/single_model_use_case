# Validation Module

## Overview

This module provides notebooks and utilities to perform **offline model validation** before a new model version is promoted to production.

## Package map

- `validate_model.py`
  - Main validation entrypoint and orchestration.
- `eval_model/`
  - `eval_model_config.py`: config-driven model evaluation implementation.
  - `eval_model_contract.py`: abstract contract for custom model evaluation.
- `eval_result/`
  - `eval_result_config.py`: config-driven evaluation result handling.
  - `eval_result_contract.py`: abstract contract for custom evaluation result handlers.

## Expected Input Parameters
```
model_type: "classifier" //Change by model
targets: "claim_status" //Change by model
target_type: "bigint" //Change by model

```

## Implementation Contracts

The module defines the following **abstract base classes** that serve as contracts. Users **must implement** all abstract methods when extending the stack:

### `EvalModelContract` (validation/eval_model/eval_model_contract.py)

| Method     | Parameters                                                                                                       | Returns                    |
| ---------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `evaluate` | `model_for_evaluate, data, targets, model_type, evaluators, evaluator_config`                                  | `EvaluationResult \| None` |

Notes:
- Built-in implementation in `eval_model_config` delegates to `mlflow.models.evaluate(...)`.
- If a custom `eval_model_module` path is passed but cannot be loaded, framework falls back to `eval_model_config`.

### `EvalResultContract` (validation/eval\_result/eval\_result\_contract.py)

| Method        | Parameters                                         | Description                    |
| ------------- | -------------------------------------------------- | ------------------------------ |
| `eval_result` | `result: EvaluationResult \| None, rules: dict`   | Custom implementation required |

Notes:
- `validate_model._evaluate_model` can return `None`.
- Contract implementations must handle `None`.
- Built-in `eval_result_config` raises `AssertionError` when `result is None`.
