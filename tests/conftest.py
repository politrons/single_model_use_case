from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


def _package(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []  # type: ignore[attr-defined]
    return mod


def _install_common_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    # ----------------------------- mlflow stubs -----------------------------
    mlflow = types.ModuleType("mlflow")

    class _Run:
        info = types.SimpleNamespace(run_id="run-id")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyModel:
        def predict(self, x):
            return [0] * len(x)

    class _DummyClient:
        def search_model_versions(self, *_args, **_kwargs):
            return []

        def get_model_version_by_alias(self, _name, _alias):
            raise Exception("alias not found")

        def get_model_version(self, _name, _version):
            return types.SimpleNamespace(aliases=[])

        def set_registered_model_alias(self, *args, **kwargs):
            return None

        def delete_registered_model_alias(self, *args, **kwargs):
            return None

    mlflow.set_experiment = lambda *a, **k: None
    mlflow.set_registry_uri = lambda *a, **k: None
    mlflow.set_tag = lambda *a, **k: None
    mlflow.end_run = lambda *a, **k: None
    mlflow.start_run = lambda *a, **k: _Run()
    mlflow.log_param = lambda *a, **k: None
    mlflow.log_metric = lambda *a, **k: None
    mlflow.log_params = lambda *a, **k: None
    mlflow.log_text = lambda *a, **k: None
    mlflow.get_experiment_by_name = lambda *_a, **_k: types.SimpleNamespace(experiment_id="1")
    mlflow.pyfunc = types.SimpleNamespace(load_model=lambda *_a, **_k: _DummyModel())
    mlflow.sklearn = types.SimpleNamespace(
        load_model=lambda *_a, **_k: _DummyModel(),
        log_model=lambda *a, **k: None,
        get_default_pip_requirements=lambda: [],
    )
    mlflow.models = types.SimpleNamespace(infer_pip_requirements=lambda **_k: [])
    mlflow.MlflowClient = _DummyClient
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)

    mlflow_models = types.ModuleType("mlflow.models")
    mlflow_models.infer_signature = lambda *a, **k: None
    mlflow_models.ModelSignature = type("ModelSignature", (), {})
    mlflow_models.EvaluationResult = type("EvaluationResult", (), {"__init__": lambda self, metrics=None: setattr(self, "metrics", metrics or {})})
    mlflow_models.evaluate = lambda *a, **k: types.SimpleNamespace(metrics={})
    monkeypatch.setitem(sys.modules, "mlflow.models", mlflow_models)

    mlflow_tracking = types.ModuleType("mlflow.tracking")
    mlflow_tracking.MlflowClient = _DummyClient
    monkeypatch.setitem(sys.modules, "mlflow.tracking", mlflow_tracking)

    # ----------------------------- pyspark stubs -----------------------------
    pyspark = types.ModuleType("pyspark")
    sql = types.ModuleType("pyspark.sql")

    class _Writer:
        def mode(self, *_a, **_k):
            return self

        def format(self, *_a, **_k):
            return self

        def option(self, *_a, **_k):
            return self

        def saveAsTable(self, *_a, **_k):
            return None

    class _SparkDf:
        @property
        def write(self):
            return _Writer()

    class _SparkSession:
        builder = types.SimpleNamespace(getOrCreate=lambda: _SparkSession())

        def createDataFrame(self, *_a, **_k):
            return _SparkDf()

    sql.SparkSession = _SparkSession
    sql.Row = lambda **kwargs: kwargs
    sql.functions = types.SimpleNamespace(col=lambda x: x, from_json=lambda c, t: c)
    sql.types = types.SimpleNamespace(MapType=lambda *a, **k: None, StringType=lambda: None, DoubleType=lambda: None)

    monkeypatch.setitem(sys.modules, "pyspark", pyspark)
    monkeypatch.setitem(sys.modules, "pyspark.sql", sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", sql.functions)
    monkeypatch.setitem(sys.modules, "pyspark.sql.types", sql.types)

    # ----------------------------- package hierarchy stubs -----------------------------
    for pkg in [
        "databricks_mlops_stack",
        "databricks_mlops_stack.training",
        "databricks_mlops_stack.training.data",
        "databricks_mlops_stack.training.model",
        "databricks_mlops_stack.split",
        "databricks_mlops_stack.utils",
        "databricks_mlops_stack.utils.constants",
        "databricks_mlops_stack.validation",
        "databricks_mlops_stack.validation.eval_model",
        "databricks_mlops_stack.validation.eval_result",
    ]:
        monkeypatch.setitem(sys.modules, pkg, _package(pkg))

    training_data_cfg = types.ModuleType("databricks_mlops_stack.training.data.training_data_config")

    class TrainingDataConfig:
        def get_training_data(self, _args):
            return pd.DataFrame({"f1": [1, 2, 3]}), pd.Series([0, 1, 0])

    training_data_cfg.TrainingDataConfig = TrainingDataConfig
    monkeypatch.setitem(sys.modules, "databricks_mlops_stack.training.data.training_data_config", training_data_cfg)

    model_cfg = types.ModuleType("databricks_mlops_stack.training.model.model_config")

    class ModelContractConfig:
        def get_model(self, _cfg):
            return types.SimpleNamespace(fit=lambda *a, **k: None, predict=lambda x: [0] * len(x))

        def log_model(self, *a, **k):
            return None

    model_cfg.ModelContractConfig = ModelContractConfig
    monkeypatch.setitem(sys.modules, "databricks_mlops_stack.training.model.model_config", model_cfg)

    predict_wrapper = types.ModuleType("databricks_mlops_stack.training.model.predict_and_proba_wrapper")

    class PredictAndProbaWrapper:
        def __init__(self, _name, internal_model):
            self.internal_model = internal_model
            self.internal_model_key = "internal"

    predict_wrapper.PredictAndProbaWrapper = PredictAndProbaWrapper
    predict_wrapper.to_prediction_series = lambda raw, _n: pd.Series(raw if isinstance(raw, (list, tuple, pd.Series)) else [raw])
    predict_wrapper.to_probability_series = lambda raw, _n: pd.Series(raw if isinstance(raw, (list, tuple, pd.Series)) else [raw])
    monkeypatch.setitem(sys.modules, "databricks_mlops_stack.training.model.predict_and_proba_wrapper", predict_wrapper)

    split_cfg = types.ModuleType("databricks_mlops_stack.split.split_config")

    class SplitContractConfig:
        def split(self, x, y, _cfg):
            return {"X_train": x, "X_test": x, "y_train": y, "y_test": y}

    split_cfg.SplitContractConfig = SplitContractConfig
    monkeypatch.setitem(sys.modules, "databricks_mlops_stack.split.split_config", split_cfg)

    core = types.ModuleType("databricks_mlops_stack.utils.constants.core")
    core.CONFIG_ENV = "env"
    core.CONFIG_RANDOM_STATE = "random_state"
    core.CONFIG_FEATURE_COLUMNS = "feature_columns"
    core.CONFIG_CREATE_VALIDATION_SET = "create_validation_set"
    core.CONFIG_DEFAULT_CATALOG_NAME = "default_catalog"
    core.CONFIG_TEMPORAL_COLUMN_NAME = "temporal_reference_column"
    core.CONFIG_TARGET_COLUMN = "target_column"
    core.X_TRAIN = "X_train"
    core.X_VAL = "X_val"
    core.X_TEST = "X_test"
    core.Y_TRAIN = "y_train"
    core.Y_VAL = "y_val"
    core.Y_TEST = "y_test"
    monkeypatch.setitem(sys.modules, "databricks_mlops_stack.utils.constants.core", core)

    model_constants = types.ModuleType("databricks_mlops_stack.utils.constants.model")
    model_constants.CONFIG_SPLIT_STRATEGY = "split_strategy"
    model_constants.CONFIG_SECTION_EARLY_STOPPING = "early_stopping"
    model_constants.CONFIG_PREDICTION_METHOD = "prediction_method"
    model_constants.PREDICTION_METHOD_PREDICT_PROBA = "predict_proba"
    model_constants.STEP_MODEL = "model"
    model_constants.CONFIG_EVAL_SET = "eval_set"
    model_constants.SPLIT_TIME = "time_series"
    model_constants.CONFIG_SECTION_DISCARDED_FEATURES = "discarded_features"
    monkeypatch.setitem(sys.modules, "databricks_mlops_stack.utils.constants.model", model_constants)

    mlops_utils = types.ModuleType("databricks_mlops_stack.utils.mlops_utils")

    class YamlUtils:
        @staticmethod
        def yaml_to_dict(_value):
            return {}

    mlops_utils.YamlUtils = YamlUtils
    monkeypatch.setitem(sys.modules, "databricks_mlops_stack.utils.mlops_utils", mlops_utils)

    mlflow_utils = types.ModuleType("databricks_mlops_stack.utils.mlflow")
    mlflow_utils.get_non_default_pipeline_params = lambda *_a, **_k: {}
    monkeypatch.setitem(sys.modules, "databricks_mlops_stack.utils.mlflow", mlflow_utils)

    eval_model_cfg = types.ModuleType("databricks_mlops_stack.validation.eval_model.eval_model_config")
    eval_model_cfg.EvalModelConfig = type("EvalModelConfig", (), {"evaluate": lambda *a, **k: types.SimpleNamespace(metrics={})})
    monkeypatch.setitem(sys.modules, "databricks_mlops_stack.validation.eval_model.eval_model_config", eval_model_cfg)

    eval_result_cfg = types.ModuleType("databricks_mlops_stack.validation.eval_result.eval_result_config")
    eval_result_cfg.EvalResultConfig = type("EvalResultConfig", (), {"eval_result": lambda *a, **k: None})
    monkeypatch.setitem(sys.modules, "databricks_mlops_stack.validation.eval_result.eval_result_config", eval_result_cfg)


@pytest.fixture
def load_entry_module(monkeypatch: pytest.MonkeyPatch):
    root = Path(__file__).resolve().parents[1]

    def _load(rel_path: str, module_name: str):
        _install_common_stubs(monkeypatch)
        path = root / rel_path
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    return _load
