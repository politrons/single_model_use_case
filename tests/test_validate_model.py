from __future__ import annotations


def test_parse_boolish_values(load_entry_module):
    mod = load_entry_module("src/validation/validate_model.py", "validate_model_test_a")
    assert mod._parse_boolish("true") is True
    assert mod._parse_boolish("0") is False
    assert mod._parse_boolish("not_bool") is None


def test_should_skip_evaluation_true(load_entry_module):
    mod = load_entry_module("src/validation/validate_model.py", "validate_model_test_b")
    cfg = mod.Config(
        env="dev",
        catalog_name="cat",
        experiment_name="exp",
        model_name="m",
        model_version=None,
        dependency_task_key=None,
        training_data_config={},
        split_config={},
        validation_config={"skip_evaluation": "true"},
        model_config={},
        eval_result_config={},
        metrics_table=None,
    )
    assert mod._should_skip_evaluation(cfg) is True


def test_should_skip_evaluation_false_when_missing(load_entry_module):
    mod = load_entry_module("src/validation/validate_model.py", "validate_model_test_c")
    cfg = mod.Config(
        env="dev",
        catalog_name="cat",
        experiment_name="exp",
        model_name="m",
        model_version=None,
        dependency_task_key=None,
        training_data_config={},
        split_config={},
        validation_config={},
        model_config={},
        eval_result_config={},
        metrics_table=None,
    )
    assert mod._should_skip_evaluation(cfg) is False

