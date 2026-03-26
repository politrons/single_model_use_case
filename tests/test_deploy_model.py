from __future__ import annotations

import sys
import types

import pytest


def test_resolve_model_uri_prefers_cli_value(load_entry_module):
    mod = load_entry_module("src/deployment/deploy_model.py", "deploy_model_test_a")
    uri = mod._resolve_model_uri("models:/cat.schema.model/5", "Training")
    assert uri == "models:/cat.schema.model/5"


def test_resolve_model_uri_from_task_values(load_entry_module, monkeypatch):
    mod = load_entry_module("src/deployment/deploy_model.py", "deploy_model_test_b")

    dbutils = types.SimpleNamespace(
        jobs=types.SimpleNamespace(
            taskValues=types.SimpleNamespace(get=lambda *_a, **_k: "models:/cat.schema.model/7")
        )
    )
    dbutils_mod = types.ModuleType("mlflow.utils.databricks_utils")
    dbutils_mod.dbutils = dbutils
    monkeypatch.setitem(sys.modules, "mlflow.utils.databricks_utils", dbutils_mod)

    uri = mod._resolve_model_uri("", "Training")
    assert uri == "models:/cat.schema.model/7"


def test_resolve_model_uri_raises_without_value(load_entry_module):
    mod = load_entry_module("src/deployment/deploy_model.py", "deploy_model_test_c")
    with pytest.raises(ValueError):
        mod._resolve_model_uri("", None)
