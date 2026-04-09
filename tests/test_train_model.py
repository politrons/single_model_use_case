from __future__ import annotations

import pandas as pd


def test_prediction_proba_to_map_from_list(load_entry_module):
    mod = load_entry_module("src/training/train_model.py", "train_model_test_a")
    result = mod._prediction_proba_to_map([0.2, 0.8])
    assert result == {"0": 0.2, "1": 0.8}


def test_prediction_proba_to_map_from_json_string(load_entry_module):
    mod = load_entry_module("src/training/train_model.py", "train_model_test_b")
    result = mod._prediction_proba_to_map('{"0": 0.1, "1": 0.9}')
    assert result == {"0": 0.1, "1": 0.9}


def test_split_predict_output_for_dataframe(load_entry_module):
    mod = load_entry_module("src/training/train_model.py", "train_model_test_c")
    raw = pd.DataFrame({"prediction": [1, 0], "prediction_proba": [0.9, 0.1]})
    preds, probs = mod._split_predict_output(raw)
    assert list(preds) == [1, 0]
    assert list(probs) == [0.9, 0.1]


def test_load_model_impl_from_contract_file(load_entry_module, tmp_path):
    mod = load_entry_module("src/training/train_model.py", "train_model_test_d")
    contract_path = tmp_path / "model_contract_impl.py"
    contract_path.write_text(
        "\n".join(
            [
                "class _Impl:",
                "    def get_model(self, args):",
                "        return args",
                "    def log_model(self, *args, **kwargs):",
                "        return None",
                "build = _Impl()",
            ]
        ),
        encoding="utf-8",
    )

    impl = mod._load_model_impl_from_contract(str(contract_path))
    assert hasattr(impl, "get_model")
    assert hasattr(impl, "log_model")


def test_load_model_impl_prefers_contract_path(load_entry_module, monkeypatch):
    mod = load_entry_module("src/training/train_model.py", "train_model_test_e")
    sentinel = object()
    monkeypatch.setattr(mod, "_load_model_impl_from_contract", lambda _p: sentinel)

    cfg = mod.Config(
        env="dev",
        catalog_name="cat",
        experiment_name="exp",
        model_name="cat.schema.model",
        databricks_mlops_stack_version="",
        training_data_config={},
        model_config={},
        model_card_path="",
        model_contract="/tmp/model_contract_impl.py",
        baseline_table_name="cat.schema.model_baseline",
    )
    assert mod._load_model_impl(cfg) is sentinel
