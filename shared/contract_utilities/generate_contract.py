"""
generate_contract.py

Utility to programmatically generate a self-contained model_contract.py that
inlines all required source files (MultiClusterWrapper, the internal model
implementation, and the contract template) into one file with a clean,
deduplicated import block at the top.

Usage example
-------------
from generate_contract import generate_model_contract

generate_model_contract(
    output_folder      = "/path/to/experiment_A",
    model_type         = "tensorflow",
    numerical_features = ["feat_1", "feat_2", "feat_3"],
    segment_columns    = ["seg_col_a", "seg_col_b", "seg_col_c"],
    config             = {"hidden_units": [64, 32]},
    random_state       = 42,
    base_params        = {"epochs": 100, "batch_size": 256, "learning_rate": 1e-3},
    extra_params       = {},
    n_jobs             = 8,
)
"""

import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Hardcoded paths — replace these with the actual locations on your system.
# ---------------------------------------------------------------------------

_MULTI_CLUSTER_WRAPPER_PATH = "contract_templates/multi_cluster_wrapper.py"

_MODEL_TYPE_TO_PATH: dict[str, str] = {
    "tensorflow": "contract_templates/ibnr_nn.py",
    "prophet": "contract_templates/inflation_prophet.py",
    ### ADD CHAIN LADDER HERE
}

_CONTRACT_TEMPLATE_PATH = "contract_templates/framework.py"

# ---------------------------------------------------------------------------
# Source-file parsing helpers
# ---------------------------------------------------------------------------

def _split_imports(source: str) -> tuple[list[str], list[str]]:
    """
    Split Python source into (import_lines, body_lines).

    Rules:
    - Only lines starting at column 0 with 'import ' or 'from ' are imports.
      This avoids hoisting imports that live inside functions or classes.
    - 'from __future__ import annotations' is dropped (not needed on 3.12+).
    - Blank lines are kept in body_lines to preserve formatting.
    """
    import_lines: list[str] = []
    body_lines: list[str] = []

    for line in source.splitlines():
        stripped = line.strip()

        if stripped == "from __future__ import annotations":
            continue

        if (line.startswith("import ") or line.startswith("from ")) and stripped:
            import_lines.append(line)
        else:
            body_lines.append(line)

    return import_lines, body_lines


def _strip_module_docstring(body_lines: list[str]) -> list[str]:
    """
    Remove the leading triple-quoted module docstring from a list of lines,
    if one is present. Returns the remaining lines.
    """
    result: list[str] = []
    in_docstring = False
    docstring_done = False

    for line in body_lines:
        stripped = line.strip()

        if not docstring_done:
            if not in_docstring and stripped.startswith('"""'):
                in_docstring = True
                # Single-line docstring: opens and closes on the same line
                if stripped.endswith('"""') and len(stripped) > 6:
                    docstring_done = True
                continue
            if in_docstring:
                if stripped.endswith('"""'):
                    docstring_done = True
                continue
            if stripped:
                docstring_done = True
                result.append(line)
        else:
            result.append(line)

    return result


def _merge_imports(import_groups: list[list[str]]) -> list[str]:
    """
    Merge multiple import-line lists into one, preserving first-seen order
    and deduplicating by exact line content.
    """
    seen: set[str] = set()
    merged: list[str] = []

    for group in import_groups:
        for line in group:
            key = line.strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(line)

    return merged


def _read_and_parse(path: str) -> tuple[list[str], list[str]]:
    """Read a .py file and return (import_lines, cleaned_body_lines)."""
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    imports, body = _split_imports(source)
    body = _strip_module_docstring(body)
    return imports, body


def _section(title: str, body_lines: list[str]) -> str:
    """Format a titled section for the output file."""
    divider = "# " + "─" * 68
    content = "\n".join(body_lines).strip()
    return f"{divider}\n# {title}\n{divider}\n{content}\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_model_contract(
    output_folder: str,
    model_type: str,
    numerical_features: list[str],
    segment_columns: list[str],
    config: dict | None = None,
    random_state: int = 42,
    base_params: dict | None = None,
    extra_params: dict | None = None,
    n_jobs: int = 8,
    relative_path: str = '../../',
) -> str:
    """
    Write a self-contained ``model_contract.py`` to ``output_folder``.

    The file is assembled by inlining, in order:
      1. MultiClusterWrapper  (multi_cluster_wrapper.py)
      2. Internal model impl  (resolved from model_type via _MODEL_TYPE_TO_PATH)
      3. Contract template    (model_contract_template.py)

    All imports are merged and deduplicated at the top of the output file.

    Parameters
    ----------
    output_folder : str
        Directory where ``model_contract.py`` will be written.
        Created automatically if it does not exist.
    model_type : str
        Key into _MODEL_TYPE_TO_PATH. Currently supported: 'tensorflow'.
    numerical_features : list[str]
        Forwarded to MultiClusterWrapper.
    segment_columns : list[str]
        Forwarded to MultiClusterWrapper.
    config : dict, optional
        Forwarded to the internal model factory via MultiClusterWrapper.
    random_state : int
        Forwarded to the internal model factory via MultiClusterWrapper.
    base_params : dict, optional
        Forwarded to the internal model factory via MultiClusterWrapper.
    extra_params : dict, optional
        Forwarded to the internal model factory via MultiClusterWrapper.
    n_jobs : int
        Parallelism for segment fitting.

    Returns
    -------
    str
        Full path to the written ``model_contract.py``.

    Raises
    ------
    ValueError
        If model_type is not in _MODEL_TYPE_TO_PATH.
    FileNotFoundError
        If any of the required source files do not exist.
    """
    config = config or {}
    base_params = base_params or {}
    extra_params = extra_params or {}

    if model_type not in _MODEL_TYPE_TO_PATH:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Known types: {list(_MODEL_TYPE_TO_PATH.keys())}"
        )

    internal_model_path = relative_path + _MODEL_TYPE_TO_PATH[model_type]
    is_tensorflow = model_type == "tensorflow"

    factory_fn_name = "_build_cluster_model"

    # ------------------------------------------------------------------
    # 1. Parse all source files
    # ------------------------------------------------------------------
    wrapper_imports, wrapper_body = _read_and_parse(relative_path + _MULTI_CLUSTER_WRAPPER_PATH)
    internal_imports, internal_body = _read_and_parse(internal_model_path)
    contract_imports, contract_body = _read_and_parse(relative_path + _CONTRACT_TEMPLATE_PATH)

    # ------------------------------------------------------------------
    # 2. Inject configuration constants into the contract body.
    #    The template is expected to contain placeholder tokens that we
    #    replace here, so the template file itself stays generic and
    #    version-controllable.
    # ------------------------------------------------------------------
    contract_body_str = "\n".join(contract_body)
    replacements = {
        "{{NUMERICAL_FEATURES}}": repr(numerical_features),
        "{{SEGMENT_COLUMNS}}":    repr(segment_columns),
        "{{CONFIG}}":             repr(config),
        "{{RANDOM_STATE}}":       repr(random_state),
        "{{BASE_PARAMS}}":        repr(base_params),
        "{{EXTRA_PARAMS}}":       repr(extra_params),
        "{{N_JOBS}}":             repr(n_jobs),
        "{{IS_TENSORFLOW}}":      repr(is_tensorflow),
        "{{FACTORY_FN_NAME}}":    factory_fn_name,
    }
    for token, value in replacements.items():
        contract_body_str = contract_body_str.replace(token, value)

    contract_body = contract_body_str.splitlines()

    # ------------------------------------------------------------------
    # 3. Merge all imports (wrapper first, then internal model, then contract)
    # ------------------------------------------------------------------
    merged_imports = _merge_imports([wrapper_imports, internal_imports, contract_imports])

    # ------------------------------------------------------------------
    # 4. Write output file
    # ------------------------------------------------------------------
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "model_contract_impl.py")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("# Auto-generated by generate_contract.py — do not edit manually.\n\n")
        fh.write("\n".join(merged_imports))
        fh.write("\n\n")
        fh.write(_section("Inlined from multi_cluster_wrapper.py", wrapper_body))
        fh.write("\n\n")
        fh.write(_section(f"Inlined from {os.path.basename(internal_model_path)}", internal_body))
        fh.write("\n\n")
        fh.write(_section("Inlined from model_contract_template.py", contract_body))
        fh.write("\n")

    return output_path 

def generate_ibnr_contract(
    model_contract_path: str,
    model_type: str,
    numerical_features: list[str],
    segment_columns: list[str],
    model_config: dict[str, Any],
    temporal_reference_column: str,
    random_state: int,
    n_jobs: int,
    relative_path: str,
) -> None:

    loss = model_config['loss']
    if loss == 'huber':
        loss = {
            'type': 'huber',
            'params': {
                'delta': 1.5,
            }
        }
    architecture = []
    for i, layer_units in enumerate(model_config['unit_per_layer']):
        architecture.append({
            'layer': {
                'type': 'dense',
                'params': {
                    'units': layer_units,
                    'activation': model_config['activation'],
                    'name': f'layer_{i}th',
                }
            }
        })

    mapping_model_contract_types = {
        'neural_network': 'tensorflow',
        #### ADD CHAIN LADDER HERE
    }
    model_contract_type = mapping_model_contract_types.get(model_type, 'unknow!!!!!!!!!!!!')
    
    generate_model_contract(
        output_folder=model_contract_path / 'training/model',
        model_type=model_contract_type,
        numerical_features=numerical_features,
        segment_columns=segment_columns,
        config={
            'architecture': architecture,
            'loss': loss,
            'optimizer': {'type': 'adam', 'params': {'learning_rate': model_config['learning_rate'],},},
            'epochs': model_config['epochs'],
            'batch_size': model_config['batch_size'],
            'eval_metric': {'type': 'mean_squared_error'},
        },
        random_state=random_state,
        base_params={'random_state': random_state},
        extra_params={
            'random_state': random_state,
            'has_hyper_search': False,
            'all_feature_transformers': ['robustscaler'],
            'number_used_features': len(numerical_features),
            'temporal_reference_column': temporal_reference_column,
            'feature_columns': numerical_features,
        },
        n_jobs=-1,
        relative_path=relative_path,
    )

    return


def generate_inflation_contract(
    model_contract_path: str | os.PathLike[str],
    clusters: list[dict[str, Any]],
    temporal_reference_column: str,
    segment_column: str,
    random_state: int,
    n_jobs: int,
    relative_path: str,
) -> dict[str, Any]:
    cluster_model_config_map: dict[str, dict[str, Any]] = {}
    regressors: set[str] = set()

    for cluster in clusters:
        raw_cluster_id = cluster.get(segment_column, cluster.get("cluster_id"))
        if raw_cluster_id is None:
            continue
        cluster_id = str(raw_cluster_id)

        model_cfg = cluster.get("model_config", {})
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        cluster_model_config_map[cluster_id] = model_cfg

        maybe_regressors = model_cfg.get("regressors")
        if isinstance(maybe_regressors, dict):
            regressors.update(str(k) for k in maybe_regressors.keys())

    default_model_config = next(iter(cluster_model_config_map.values()), {})
    feature_columns = [temporal_reference_column] + sorted(regressors) + [segment_column]

    generate_model_contract(
        output_folder=Path(model_contract_path) / "training/model",
        model_type="prophet",
        numerical_features=feature_columns,
        segment_columns=[segment_column],
        config={
            "default_model_config": default_model_config,
            "cluster_model_config_map": cluster_model_config_map,
        },
        random_state=random_state,
        base_params={"random_state": random_state},
        extra_params={
            "random_state": random_state,
            "feature_columns": feature_columns,
            "temporal_reference_column": temporal_reference_column,
            "segment_column": segment_column,
        },
        n_jobs=n_jobs,
        relative_path=relative_path,
    )

    return {
        "feature_columns": feature_columns,
        "segment_columns": [segment_column],
        "regressor_columns": sorted(regressors),
        "cluster_model_config_map": cluster_model_config_map,
    }


def _build_factory_resource_content(model_name: str) -> str:
    workspace_files = "${var.workspace_name}/${bundle.name}/${bundle.target}/files"
    experiment_name = f"/Users/${{workspace.current_user.userName}}/${{bundle.target}}-{model_name}_exp"
    model_uc_name = f"${{var.catalog_name}}.${{bundle.name}}.{model_name}"
    configs_root = f"{workspace_files}/configs/{model_name}"
    contract_path = f"{workspace_files}/contracts/{model_name}/training/model/model_contract_impl.py"
    metrics_table = f"${{var.catalog_name}}.${{bundle.name}}.{model_name}_metrics"
    baseline_table = f"${{var.catalog_name}}.${{bundle.name}}.{model_name}_baseline"

    return f"""common_permissions: &permissions
  permissions:
    - level: CAN_MANAGE
      group_name: users

resources:
  jobs:
    {model_name}-factory-job:
      name: ${{bundle.target}}-${{bundle.name}}-{model_name}-factory-job
      environments:
        - environment_key: serverless_default
          spec:
            environment_version: "4"
            dependencies:
              - "imbalanced-learn==0.14.0"
              - "lightgbm==4.3.0"
              - "mlflow==3.0.1"
              - "neuralprophet==0.8.0"
              - "prophet==1.1.6"
              - "numpy==1.26.4"
              - "optuna==3.6.0"
              - "optuna-integration==3.6.0"
              - "pandas==2.3.3"
              - "ray[all]"
              - "scikit-learn==1.6.1"
              - "scipy==1.16.3"
              - "torch==2.5.1"
              - "tensorflow==2.17.0"
              - "xgboost==2.1.4"
              - "pyspark"
              - "PyYAML"
              - "pytz~=2022.2.1"
              - "databricks-sdk>=0.57"
              - "databricks-feature-engineering==0.12.1"
              - "azure-keyvault==4.2.0"
              - "shap==0.46.0"
      tasks:
        - task_key: Training
          environment_key: serverless_default
          spark_python_task:
            python_file: "{workspace_files}/src/training/train_model.py"
            source: WORKSPACE
            parameters:
              - "--env"
              - "${{bundle.target}}"
              - "--experiment_name"
              - "{experiment_name}"
              - "--model_name"
              - "{model_uc_name}"
              - "--catalog_name"
              - "${{var.catalog_name}}"
              - "--training_data_config"
              - "{configs_root}/training_data_config.yml"
              - "--model_config"
              - "{configs_root}/model_config.yml"
              - "--model_contract"
              - "{contract_path}"
              - "--split_config"
              - "{configs_root}/split_config.yml"
              - "--metrics_latency_table"
              - "{metrics_table}"
              - "--validation_config"
              - "{configs_root}/validation_config.yml"
              - "--baseline_table_name"
              - "{baseline_table}"
        - task_key: Validation
          environment_key: serverless_default
          depends_on:
            - task_key: Training
          spark_python_task:
            python_file: "{workspace_files}/src/validation/validate_model.py"
            source: WORKSPACE
            parameters:
              - "--env"
              - "${{bundle.target}}"
              - "--dependency_task_key"
              - "Training"
              - "--catalog_name"
              - "${{var.catalog_name}}"
              - "--model_name"
              - "{model_uc_name}"
              - "--experiment_name"
              - "{experiment_name}"
              - "--training_data_config"
              - "{configs_root}/training_data_config.yml"
              - "--model_config"
              - "{configs_root}/model_config.yml"
              - "--validation_config"
              - "{configs_root}/validation_config.yml"
              - "--eval_result_config"
              - "{configs_root}/eval_result_config.yml"
              - "--split_config"
              - "{configs_root}/split_config.yml"
              - "--metrics_table"
              - "{metrics_table}"
        - task_key: Deployment
          environment_key: serverless_default
          depends_on:
            - task_key: Validation
          spark_python_task:
            python_file: "{workspace_files}/src/deployment/deploy_model.py"
            source: WORKSPACE
            parameters:
              - "--env"
              - "${{bundle.target}}"
      <<: *permissions
"""


def _build_batch_resource_content(model_name: str) -> str:
    workspace_files = "${var.workspace_name}/${bundle.name}/${bundle.target}/files"
    experiment_name = f"/Users/${{workspace.current_user.userName}}/${{bundle.target}}-{model_name}_exp"
    model_uc_name = f"${{var.catalog_name}}.${{bundle.name}}.{model_name}"
    configs_root = f"{workspace_files}/configs/{model_name}"
    contract_path = f"{workspace_files}/contracts/{model_name}/training/model/model_contract_impl.py"
    output_table = f"${{var.catalog_name}}.${{bundle.name}}.{model_name}_batch_predictions"

    return f"""common_permissions: &permissions
  permissions:
    - level: CAN_MANAGE
      group_name: users

resources:
  jobs:
    {model_name}-batch-job:
      name: ${{bundle.target}}-${{bundle.name}}-{model_name}-batch-job
      job_clusters:
        - job_cluster_key: Job_cluster
          new_cluster:
            spark_version: 16.4.x-scala2.12
            instance_pool_id: ${{var.instance_pool_id}}
            driver_instance_pool_id: ${{var.instance_pool_id}}
            policy_id: ${{var.policy_id}}
            custom_tags:
              local-dbx-OpsTeam: CDAO ML Engineering
              local-dbx-opco: ${{var.load_dbx_opco}}
              local-dbx-env: ${{var.job_cluster_env}}
              local-dbx-domain: CDAO
            data_security_mode: SINGLE_USER
            runtime_engine: STANDARD
            kind: CLASSIC_PREVIEW
            use_ml_runtime: true
            is_single_node: false
            autoscale:
              min_workers: 1
              max_workers: 2
      tasks:
        - task_key: Batch
          job_cluster_key: Job_cluster
          libraries:
            - pypi:
                package: mlflow==3.0.1
            - pypi:
                package: pandas==2.3.3
            - pypi:
                package: scikit-learn==1.6.1
            - pypi:
                package: prophet==1.1.6
            - pypi:
                package: PyYAML
          spark_python_task:
            python_file: "{workspace_files}/src/batch/batch_model.py"
            source: WORKSPACE
            parameters:
              - "--env"
              - "${{bundle.target}}"
              - "--catalog_name"
              - "${{var.catalog_name}}"
              - "--model_name"
              - "{model_uc_name}"
              - "--experiment_name"
              - "{experiment_name}"
              - "--model_alias"
              - "champion"
              - "--training_data_config"
              - "{configs_root}/training_data_config.yml"
              - "--split_config"
              - "{configs_root}/split_config.yml"
              - "--model_contract"
              - "{contract_path}"
              - "--output_table"
              - "{output_table}"
      <<: *permissions
"""


def generate_factory_and_batch_resources(
    resources_path: str | os.PathLike[str],
    model_name: str,
) -> tuple[Path, Path]:
    resources_dir = Path(resources_path)
    resources_dir.mkdir(parents=True, exist_ok=True)

    factory_path = resources_dir / f"factory-{model_name}-resource.yml"
    batch_path = resources_dir / f"batch-{model_name}-resource.yml"

    factory_path.write_text(_build_factory_resource_content(model_name), encoding="utf-8")
    batch_path.write_text(_build_batch_resource_content(model_name), encoding="utf-8")

    return factory_path, batch_path


def update_bundle_include(
    bundle_file_path: str | os.PathLike[str],
    resource_paths: list[str | os.PathLike[str]],
) -> list[str]:
    """Replace databricks.yml include entries with the provided resource files."""
    bundle_path = Path(bundle_file_path)
    text = bundle_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    include_entries = [f"./resources/{Path(p).name}" for p in resource_paths]
    include_entries = sorted(dict.fromkeys(include_entries))
    include_block = ["include:\n"] + [f"  - {entry}\n" for entry in include_entries]

    include_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "include:":
            include_idx = i
            break

    if include_idx is None:
        insert_at = len(lines)
        for i, line in enumerate(lines):
            if line.strip().startswith("targets:"):
                insert_at = i
                break
        if insert_at > 0 and lines[insert_at - 1].strip() != "":
            include_block = ["\n"] + include_block
        lines[insert_at:insert_at] = include_block
    else:
        include_indent = len(lines[include_idx]) - len(lines[include_idx].lstrip(" "))
        end_idx = include_idx + 1
        while end_idx < len(lines):
            current = lines[end_idx]
            stripped = current.strip()
            current_indent = len(current) - len(current.lstrip(" "))
            if stripped and current_indent <= include_indent and not stripped.startswith("-"):
                break
            end_idx += 1
        lines[include_idx:end_idx] = include_block

    bundle_path.write_text("".join(lines), encoding="utf-8")
    return include_entries
