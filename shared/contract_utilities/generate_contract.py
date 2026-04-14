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
    "chain_ladder": "contract_templates/ibnr_cl.py",
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
    relative_path: str = "../../",
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
        raise ValueError(f"Unknown model_type '{model_type}'. Known types: {list(_MODEL_TYPE_TO_PATH.keys())}")

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
        "{{SEGMENT_COLUMNS}}": repr(segment_columns),
        "{{CONFIG}}": repr(config),
        "{{RANDOM_STATE}}": repr(random_state),
        "{{BASE_PARAMS}}": repr(base_params),
        "{{EXTRA_PARAMS}}": repr(extra_params),
        "{{N_JOBS}}": repr(n_jobs),
        "{{IS_TENSORFLOW}}": repr(is_tensorflow),
        "{{FACTORY_FN_NAME}}": factory_fn_name,
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
    model_config = model_config or {}

    mapping_model_contract_types = {"neural_network": "tensorflow", "chain_ladder": "chain_ladder"}
    model_contract_type = mapping_model_contract_types.get(model_type, "unknow!!!!!!!!!!!!")

    if model_type == "chain_ladder":
        occurrence_date_col = str(model_config.get("occurrence_date_col") or temporal_reference_column)
        lag_col = str(model_config.get("lag_col") or "lag")
        chain_ladder_input_columns = list(dict.fromkeys([*numerical_features, occurrence_date_col, lag_col]))

        generate_model_contract(
            output_folder=model_contract_path / "training/model",
            model_type=model_contract_type,
            numerical_features=chain_ladder_input_columns,
            segment_columns=segment_columns,
            config={
                "occurrence_date_col": occurrence_date_col,
                "lag_col": lag_col,
            },
            random_state=random_state,
            base_params={"random_state": random_state},
            extra_params={
                "random_state": random_state,
                "temporal_reference_column": temporal_reference_column,
                "feature_columns": chain_ladder_input_columns,
            },
            n_jobs=-1,
            relative_path=relative_path,
        )

    else:
        loss = model_config["loss"]
        if loss == "huber":
            loss = {
                "type": "huber",
                "params": {
                    "delta": 1.5,
                },
            }
        architecture = []
        for i, layer_units in enumerate(model_config["unit_per_layer"]):
            architecture.append(
                {
                    "layer": {
                        "type": "dense",
                        "params": {
                            "units": layer_units,
                            "activation": model_config["activation"],
                            "name": f"layer_{i}th",
                        },
                    }
                }
            )

        generate_model_contract(
            output_folder=model_contract_path / "training/model",
            model_type=model_contract_type,
            numerical_features=numerical_features,
            segment_columns=segment_columns,
            config={
                "architecture": architecture,
                "loss": loss,
                "optimizer": {
                    "type": "adam",
                    "params": {
                        "learning_rate": model_config["learning_rate"],
                    },
                },
                "epochs": model_config["epochs"],
                "batch_size": model_config["batch_size"],
                "eval_metric": {"type": "mean_squared_error"},
            },
            random_state=random_state,
            base_params={"random_state": random_state},
            extra_params={
                "random_state": random_state,
                "has_hyper_search": False,
                "all_feature_transformers": ["robustscaler"],
                "number_used_features": len(numerical_features),
                "temporal_reference_column": temporal_reference_column,
                "feature_columns": numerical_features,
            },
            n_jobs=-1,
            relative_path=relative_path,
        )

    return


def generate_inflation_contract(
    model_contract_path: str | os.PathLike[str],
    temporal_reference_column: str,
    segment_column: str,
    random_state: int,
    n_jobs: int,
    relative_path: str,
    cluster_model_config_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    resolved_cluster_model_config_map: dict[str, dict[str, Any]] = {}
    regressors: set[str] = set()

    for raw_cluster_id, raw_model_cfg in cluster_model_config_map.items():
        model_cfg = raw_model_cfg if isinstance(raw_model_cfg, dict) else {}
        resolved_cluster_model_config_map[str(raw_cluster_id)] = model_cfg

    for model_cfg in resolved_cluster_model_config_map.values():
        maybe_regressors = model_cfg.get("regressors")
        if isinstance(maybe_regressors, dict):
            regressors.update(str(k) for k in maybe_regressors.keys())

    default_model_config = next(iter(resolved_cluster_model_config_map.values()), {})
    feature_columns = [temporal_reference_column] + sorted(regressors) + [segment_column]

    generate_model_contract(
        output_folder=Path(model_contract_path) / "training/model",
        model_type="prophet",
        numerical_features=feature_columns,
        segment_columns=[segment_column],
        config={
            "default_model_config": default_model_config,
            "cluster_model_config_map": resolved_cluster_model_config_map,
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
        "cluster_model_config_map": resolved_cluster_model_config_map,
    }
