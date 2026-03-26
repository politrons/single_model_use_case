from __future__ import annotations

import types

import pandas as pd


def test_split_predict_output_with_dict(load_entry_module):
    mod = load_entry_module("src/batch/batch_model.py", "batch_model_test_a")
    preds, probs = mod._split_predict_output({"prediction": [1, 0], "prediction_proba": [0.8, 0.2]})
    assert preds == [1, 0]
    assert probs == [0.8, 0.2]


def test_resolve_model_uri_uses_direct_value(load_entry_module):
    mod = load_entry_module("src/batch/batch_model.py", "batch_model_test_b")
    cfg = mod.Config(
        env="dev",
        catalog_name="cat",
        model_name="cat.schema.model",
        output_table="cat.schema.out",
        training_data_config={},
        split_config={},
        model_uri="models:/cat.schema.model/11",
    )
    assert mod._resolve_model_uri(cfg) == "models:/cat.schema.model/11"


def test_latest_model_version_chooses_max(load_entry_module, monkeypatch):
    mod = load_entry_module("src/batch/batch_model.py", "batch_model_test_c")

    class _Client:
        def search_model_versions(self, _query):
            return [types.SimpleNamespace(version="1"), types.SimpleNamespace(version="3"), types.SimpleNamespace(version="2")]

    monkeypatch.setattr(mod, "MlflowClient", _Client)
    assert mod._latest_model_version("cat.schema.model") == 3


def test_build_predictions_df_adds_prediction_columns(load_entry_module):
    mod = load_entry_module("src/batch/batch_model.py", "batch_model_test_d")

    class _Model:
        def predict(self, x):
            return pd.DataFrame({"prediction": [1] * len(x), "prediction_proba": [0.9] * len(x)})

    cfg = mod.Config(
        env="dev",
        catalog_name="cat",
        model_name="cat.schema.model",
        output_table="cat.schema.out",
        training_data_config={},
        split_config={},
    )
    x_df = pd.DataFrame({"f1": [10, 20]})
    y_df = pd.Series([0, 1])
    out = mod._build_predictions_df(x_df, y_df, "X_test", _Model(), "models:/cat.schema.model/1", cfg)
    assert "prediction" in out.columns
    assert "prediction_proba" in out.columns
    assert "split_name" in out.columns
    assert len(out) == 2


def test_load_model_with_compat_installs_shim_then_loads(load_entry_module, monkeypatch):
    mod = load_entry_module("src/batch/batch_model.py", "batch_model_test_e")

    calls: list[str] = []
    expected = object()

    def _fake_install():
        calls.append("install")
        return True

    def _fake_load_model(_uri):
        calls.append("load")
        return expected

    monkeypatch.setattr(mod, "_install_sklearn_pickle_compat", _fake_install)
    monkeypatch.setattr(mod.mlflow.pyfunc, "load_model", _fake_load_model)
    out = mod._load_model_with_compat("models:/cat.schema.model/3")
    assert out is expected
    assert calls == ["install", "load"]


def test_load_model_with_compat_propagates_load_error(load_entry_module, monkeypatch):
    mod = load_entry_module("src/batch/batch_model.py", "batch_model_test_f")

    def _fake_load_model(_uri):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod.mlflow.pyfunc, "load_model", _fake_load_model)
    monkeypatch.setattr(mod, "_install_sklearn_pickle_compat", lambda: True)

    try:
        mod._load_model_with_compat("models:/cat.schema.model/3")
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert str(exc) == "boom"
