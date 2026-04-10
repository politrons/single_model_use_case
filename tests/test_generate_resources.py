from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_generate_factory_and_batch_resources_creates_expected_files(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "shared" / "contract_utilities" / "generate_contract.py"
    spec = importlib.util.spec_from_file_location("generate_contract_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_contract_test_module"] = module
    spec.loader.exec_module(module)

    generate_factory_and_batch_resources = module.generate_factory_and_batch_resources

    model_name = "ibnr_abc123"
    factory_path, batch_path = generate_factory_and_batch_resources(tmp_path, model_name)

    assert factory_path.name == f"factory-{model_name}-resource.yml"
    assert batch_path.name == f"batch-{model_name}-resource.yml"
    assert factory_path.is_file()
    assert batch_path.is_file()

    factory_text = factory_path.read_text(encoding="utf-8")
    batch_text = batch_path.read_text(encoding="utf-8")

    assert f"{model_name}-factory-job" in factory_text
    assert f"configs/{model_name}/training_data_config.yml" in factory_text
    assert f"contracts/{model_name}/training/model/model_contract_impl.py" in factory_text
    assert "prophet==1.1.6" in factory_text

    assert f"{model_name}-batch-job" in batch_text
    assert f"configs/{model_name}/split_config.yml" in batch_text
    assert f"contracts/{model_name}/training/model/model_contract_impl.py" in batch_text
    assert "prophet==1.1.6" in batch_text


def test_update_bundle_include_replaces_existing_entries(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "shared" / "contract_utilities" / "generate_contract.py"
    spec = importlib.util.spec_from_file_location("generate_contract_test_module_2", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_contract_test_module_2"] = module
    spec.loader.exec_module(module)

    bundle_path = tmp_path / "databricks.yml"
    bundle_path.write_text(
        "\n".join(
            [
                "bundle:",
                "  name: demo",
                "include:",
                "  - ./resources/old-a.yml",
                "  - ./resources/old-b.yml",
                "targets:",
                "  dev:",
                "    mode: development",
                "",
            ]
        ),
        encoding="utf-8",
    )

    include = module.update_bundle_include(
        bundle_file_path=bundle_path,
        resource_paths=[
            tmp_path / "factory-m1-resource.yml",
            tmp_path / "batch-m1-resource.yml",
        ],
    )
    content = bundle_path.read_text(encoding="utf-8")

    assert include == [
        "./resources/batch-m1-resource.yml",
        "./resources/factory-m1-resource.yml",
    ]
    assert "./resources/old-a.yml" not in content
    assert "./resources/old-b.yml" not in content
    assert "include:\n  - ./resources/batch-m1-resource.yml\n  - ./resources/factory-m1-resource.yml\n" in content


def test_generate_inflation_contract_creates_model_contract_file(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "shared" / "contract_utilities" / "generate_contract.py"
    spec = importlib.util.spec_from_file_location("generate_contract_test_module_3", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_contract_test_module_3"] = module
    spec.loader.exec_module(module)

    model_contract_path = tmp_path / "inflation_bc"
    result = module.generate_inflation_contract(
        model_contract_path=model_contract_path,
        clusters=[
            {
                "cluster_id": "cluster_1",
                "model_config": {
                    "regressors": {"price_CPI": 10},
                    "changepoint_prior_scale": 1.5,
                },
            },
            {
                "cluster_id": "cluster_2",
                "model_config": {
                    "regressors": None,
                    "changepoint_prior_scale": 2.5,
                },
            },
        ],
        temporal_reference_column="YearMonthDate",
        segment_column="cluster_id",
        random_state=42,
        n_jobs=-1,
        relative_path="",
    )

    generated_contract = model_contract_path / "training" / "model" / "model_contract_impl.py"
    assert generated_contract.is_file()
    content = generated_contract.read_text(encoding="utf-8")

    assert "from prophet import Prophet" in content
    assert "cluster_model_config_map" in content
    assert "default_model_config" in content
    assert "group[self.numerical_features].values" not in content
    assert "group[self.numerical_features]," in content
    assert "X = group[self.numerical_features]" in content
    assert "from tensorflow import keras  # type: ignore[import-untyped]" in content
    assert "import tensorflow as tf  # type: ignore[import-untyped]" in content
    assert 'if step_model is not None and hasattr(step_model, "prepare_for_serialization"):' in content
    assert 'if nested_model is not None and hasattr(nested_model, "prepare_for_serialization"):' in content
    assert set(result["feature_columns"]) == {"YearMonthDate", "price_CPI", "cluster_id"}
