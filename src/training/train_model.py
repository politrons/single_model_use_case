from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow  # type: ignore # noqa
import pandas as pd  # type: ignore # noqa
from mlflow import MlflowClient  # type: ignore # noqa
from mlflow.models import infer_signature, ModelSignature  # type: ignore # noqa
from pyspark.sql import functions as F, types as T  # type: ignore # noqa
from pyspark.sql import SparkSession  # type: ignore # noqa

# Allow running this script directly from workspace source without installing a wheel.
_THIS_FILE = globals().get("__file__") or (
    inspect.currentframe().f_code.co_filename if inspect.currentframe() else ""
)
if not _THIS_FILE:
    raise RuntimeError("Could not resolve script file path to initialize source imports.")
_SRC_ROOT = Path(_THIS_FILE).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if "databricks_mlops_stack" not in sys.modules:
    import types

    _pkg = types.ModuleType("databricks_mlops_stack")
    _pkg.__path__ = [str(_SRC_ROOT)]  # type: ignore[attr-defined]
    _pkg.__file__ = str(_SRC_ROOT / "__init__.py")
    sys.modules["databricks_mlops_stack"] = _pkg

from databricks_mlops_stack.training.data.training_data_config import TrainingDataConfig  # type: ignore # noqa
from databricks_mlops_stack.training.model.model_config import ModelContractConfig  # type: ignore # noqa
from databricks_mlops_stack.training.model.predict_and_proba_wrapper import (  # type: ignore # noqa
    PredictAndProbaWrapper,
    to_prediction_series as _to_prediction_series,
    to_probability_series as _to_probability_series,
)
from databricks_mlops_stack.split.split_config import SplitContractConfig  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.model import (  # type: ignore # noqa
    CONFIG_SPLIT_STRATEGY,
    CONFIG_SECTION_EARLY_STOPPING,
    CONFIG_PREDICTION_METHOD,
    PREDICTION_METHOD_PREDICT_PROBA,
    STEP_MODEL,
    CONFIG_EVAL_SET,
    SPLIT_TIME,
    CONFIG_SECTION_DISCARDED_FEATURES,
)
from databricks_mlops_stack.utils.constants.core import (  # type: ignore # noqa
    CONFIG_ENV,
    CONFIG_RANDOM_STATE,
    CONFIG_FEATURE_COLUMNS,
    X_TRAIN,
    X_VAL,
    X_TEST,
    Y_TRAIN,
    Y_VAL,
    Y_TEST,
    CONFIG_CREATE_VALIDATION_SET,
    CONFIG_DEFAULT_CATALOG_NAME,
    CONFIG_TEMPORAL_COLUMN_NAME,
)
from databricks_mlops_stack.utils.mlops_utils import YamlUtils  # type: ignore # noqa
from databricks_mlops_stack.utils.mlflow import get_non_default_pipeline_params  # type: ignore # noqa

# ----------------------------- logging -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)

LOG = logging.getLogger("framework.train")


# ----------------------------- config -----------------------------
@dataclass(frozen=True)
class Config:
    env: str
    catalog_name: str
    experiment_name: str
    model_name: str
    databricks_mlops_stack_version: str
    training_data_config: dict[str, Any]
    model_config: dict[str, Any]
    model_card_path: str
    metrics_latency_table: str = ""
    quality_monitor_config: dict[str, Any] = field(default_factory=dict)
    validation_config: dict[str, Any] = field(default_factory=dict)
    baseline_table_name: str = ""
    target_alias: str = "challenger"
    split_config: dict[str, Any] = field(default_factory=dict)


# ----------------------------- helpers -----------------------------

def _resolve_prediction_method(cfg: Config) -> str:
    raw = (cfg.model_config or {}).get(CONFIG_PREDICTION_METHOD, "")
    return str(raw).strip().lower()


def _maybe_wrap_model_for_prediction_method(cfg: Config, model: Any) -> Any:
    prediction_method = _resolve_prediction_method(cfg)
    if prediction_method != PREDICTION_METHOD_PREDICT_PROBA:
        LOG.info("No predict method prediction_prob detected. Using regular model ")
        return model
    LOG.info("Predict method prediction_prob detected. Using Custom model ")
    return PredictAndProbaWrapper(cfg.model_name, model)


def _model_for_param_logging(model: Any) -> Any:
    if isinstance(model, PredictAndProbaWrapper):
        return model.internal_model
    return model


def _split_predict_output(raw_output: Any) -> tuple[Any, Any | None]:
    """Split model.predict output into (prediction, probability?)."""
    LOG.info("Splitting model.predict output. raw_output_type=%s", type(raw_output).__name__)
    if isinstance(raw_output, pd.DataFrame):
        lower_cols = {str(c).lower(): c for c in raw_output.columns}
        pred_col = lower_cols.get("prediction")
        if pred_col is None:
            pred_col = raw_output.columns[0] if len(raw_output.columns) > 0 else None
        prob_col = (
            lower_cols.get("prediction_proba")
            or lower_cols.get("probability")
            or lower_cols.get("probabilities")
            or lower_cols.get("proba")
            or lower_cols.get("predict_proba")
        )
        if prob_col is None and pred_col is not None and len(raw_output.columns) == 2:
            remaining = [c for c in raw_output.columns if c != pred_col]
            prob_col = remaining[0] if remaining else None
        if prob_col is None and pred_col is None and len(raw_output.columns) == 2:
            pred_col = raw_output.columns[0]
            prob_col = raw_output.columns[1]
        preds = raw_output[pred_col] if pred_col is not None else raw_output
        probs = raw_output[prob_col] if prob_col is not None else None
        if probs is None:
            LOG.info(
                "No probability column detected in DataFrame output. "
                "Looked for: prediction_proba/probability/probabilities/proba/predict_proba. "
                "Columns=%s",
                list(raw_output.columns),
            )
        else:
            LOG.info("Probability column detected in DataFrame output: %s", prob_col)
        return preds, probs

    if isinstance(raw_output, dict):
        preds = raw_output.get("prediction", raw_output.get("predictions"))
        probs = raw_output.get(
            "prediction_proba",
            raw_output.get("probability", raw_output.get("probabilities", raw_output.get("proba"))),
        )
        if preds is not None:
            if probs is None:
                LOG.info(
                    "No probability key detected in dict output. "
                    "Looked for: prediction_proba/probability/probabilities/proba. "
                    "Keys=%s",
                    list(raw_output.keys()),
                )
            else:
                LOG.info("Probability key detected in dict output.")
            return preds, probs

    if isinstance(raw_output, tuple) and len(raw_output) == 2:
        LOG.info("Tuple(prediction, probability) detected in predict output.")
        return raw_output[0], raw_output[1]

    LOG.info(
        "Predict output does not include a probability structure. "
        "Expected DataFrame/dict/tuple(2). raw_output_type=%s",
        type(raw_output).__name__,
    )
    return raw_output, None


def _load_model_impl(cfg):
    LOG.info("Loading model from config....")
    return ModelContractConfig()


def _get_metadata(
        cfg: Config,
        all_split_set: dict[str, Any],
    ) -> dict[str, Any]:

    extra_params = {}
    maybe_early_stop = cfg.model_config.get(CONFIG_SECTION_EARLY_STOPPING)
    if maybe_early_stop:
        if X_VAL not in all_split_set:
            error_msg = f"When specifying '{CONFIG_SECTION_EARLY_STOPPING}', "
            error_msg += f"it is required to set '{CONFIG_CREATE_VALIDATION_SET}' as True"
            raise ValueError(error_msg)
        extra_params = {f'{STEP_MODEL}__{CONFIG_EVAL_SET}': [(all_split_set[X_VAL], all_split_set[Y_VAL])]}

    return extra_params


def _time_fit(
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: dict[str, Any],
    ) -> tuple[Any, float]:
    LOG.info(f"Train set shape: {X.shape}")
    t0 = time.perf_counter()
    model.fit(X, y, **metadata)
    t1 = time.perf_counter()
    return model, (t1 - t0) * 1000.0


def _time_predict(model: Any, x: pd.DataFrame) -> tuple[Any, float]:
    t0 = time.perf_counter()
    preds = model.predict(x)
    t1 = time.perf_counter()
    return preds, (t1 - t0) * 1000.0


def _infer_signature(model: Any, x_train: pd.DataFrame):
    sample_in = x_train.iloc[: min(100, len(x_train))].copy()
    raw_sample_out = model.predict(sample_in)
    pred_values, proba_values = _split_predict_output(raw_sample_out)
    sample_out_df = pd.DataFrame({"prediction": _to_prediction_series(pred_values, len(sample_in))})
    if proba_values is not None:
        sample_out_df["prediction_proba"] = _to_probability_series(proba_values, len(sample_in))
    signature = infer_signature(sample_in, sample_out_df)
    input_example = sample_in.head(5)
    return signature, input_example


def _latest_model_version(name: str) -> int:
    client = MlflowClient()
    latest = 0
    for mv in client.search_model_versions(f"name='{name}'"):
        try:
            v = int(mv.version)
            latest = max(latest, v)
        except Exception:
            continue
    return latest


def _set_task_values(model_uri: str, model_name: str, model_version: int) -> None:
    """Best-effort: expose values to downstream tasks when running inside Databricks."""
    try:
        from mlflow.utils.databricks_utils import dbutils  # type: ignore
        dbutils.jobs.taskValues.set("model_uri", model_uri)
        dbutils.jobs.taskValues.set("model_name", model_name)
        dbutils.jobs.taskValues.set("model_version", model_version)
    except Exception as exc:
        LOG.info("dbutils.taskValues.set not available: %s", exc)


def _append_latency_metrics(
        spark: SparkSession,
        cfg: Config,
        run_id: str,
        model_version: int,
        train_ms: float,
        predict_ms: float,
) -> None:
    if not cfg.metrics_latency_table:
        return
    from pyspark.sql import Row
    from datetime import datetime, timezone

    exp = mlflow.get_experiment_by_name(cfg.experiment_name)
    experiment_id = exp.experiment_id if exp else None

    base = dict(
        ts=datetime.now(timezone.utc),
        env=cfg.env,
        experiment_id=experiment_id,
        experiment_name=cfg.experiment_name,
        run_id=run_id,
        model_name=cfg.model_name,
        model_version=int(model_version),
    )
    metrics = {
        "train_time_ms": float(train_ms),
        "predict_time_ms": float(predict_ms),
    }
    rows = [Row(**base, metric_name=k, metric_value=v) for k, v in metrics.items()]
    (
        spark.createDataFrame(rows)
        .write
        .mode("append")
        .format("delta")
        .option("mergeSchema", "true")
        .saveAsTable(cfg.metrics_latency_table)
    )
    LOG.info("Appended %d metrics rows to %s", len(rows), cfg.metrics_latency_table)


def _register_model(cfg: Config,
                    input_example,
                    model,
                    model_impl: ModelContractConfig | Any,
                    predict_ms: float,
                    signature: ModelSignature,
                    train_ms: float) -> str:
    with mlflow.start_run(run_name=f"train_{cfg.env}") as run:
        run_id = run.info.run_id
        mlflow.log_param("env", cfg.env)
        mlflow.log_param("feature_engineering_inline", True)
        mlflow.log_metric("train_time_ms", float(train_ms))
        mlflow.log_metric("predict_time_ms", float(predict_ms))
        mlflow.log_params(get_non_default_pipeline_params(_model_for_param_logging(model)))
        model_impl.log_model(model, cfg.model_name, signature, input_example, cfg.model_config)
    return run_id


def _create_baseline_table(
        all_split_set,
        cfg: Config,
        preds,
        prediction_proba,
        spark: SparkSession,
        version: int,
):
    if cfg.quality_monitor_config.get("baseline_enable") and cfg.quality_monitor_config["baseline_enable"] == "True":
        # Idempotent creation
        x_train = all_split_set[X_TEST].copy()
        x_train["prediction"] = _to_prediction_series(preds, len(x_train))
        if prediction_proba is not None:
            x_train["prediction_proba"] = _to_probability_series(prediction_proba, len(x_train))
            x_train["prediction_proba"] = x_train["prediction_proba"].map(_prediction_proba_to_map_json)
        x_train["label_col"] = all_split_set[Y_TEST]
        x_train["model_id_col"] = f"{cfg.model_name}-{version}"
        baseline_df = spark.createDataFrame(x_train)
        targets_type = cfg.validation_config.get("target_type")
        baseline_df = baseline_df.withColumn("label_col", F.col("label_col").cast(targets_type))
        baseline_df = baseline_df.withColumn("prediction", F.col("prediction").cast(targets_type))
        if prediction_proba is not None:
            baseline_df = baseline_df.withColumn(
                "prediction_proba",
                F.from_json(F.col("prediction_proba"), T.MapType(T.StringType(), T.DoubleType())),
            )
        (
            baseline_df.write
            .format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(cfg.baseline_table_name)
        )
        LOG.info(f"Baseline table: new {baseline_df.count()} rows append for model_id_col={cfg.model_name}-{version}")


def _prediction_proba_to_map_json(raw_value: Any) -> str | None:
    prob_map = _prediction_proba_to_map(raw_value)
    if prob_map is None:
        return None
    return json.dumps(prob_map)


def _prediction_proba_to_map(raw_value: Any) -> dict[str, float] | None:
    if raw_value is None:
        return None

    try:
        maybe_na = pd.isna(raw_value)
        if isinstance(maybe_na, bool) and maybe_na:
            return None
    except Exception:
        pass

    if isinstance(raw_value, dict):
        return {str(k): float(v) for k, v in raw_value.items()}

    if isinstance(raw_value, str):
        s = raw_value.strip()
        if not s:
            return None
        if s.startswith("{") or s.startswith("["):
            try:
                parsed = json.loads(s)
            except Exception:
                return None
            return _prediction_proba_to_map(parsed)
        try:
            return {"0": float(s)}
        except Exception:
            return None

    if isinstance(raw_value, (list, tuple)):
        return {str(i): float(v) for i, v in enumerate(raw_value)}

    return {"0": float(raw_value)}

def _register_model_card(cfg: Config):
    if cfg.model_card_path:
        p = Path(cfg.model_card_path)
        if p.is_file():
            try:
                MlflowClient().update_registered_model(
                    name=cfg.model_name, description=p.read_text(encoding="utf-8")
                )
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Could not update model card: %s", exc)

def _tag_framework_version(cfg: Config):
    try:
        MlflowClient().set_registered_model_tag(
            name=cfg.model_name, key="databricks_mlops_stack_version", value=str(cfg.databricks_mlops_stack_version)
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Could not set tag '%s': %s", cfg.databricks_mlops_stack_version, exc)


def _register_model_alias(cfg: Config, version: int):
    try:
        MlflowClient().set_registered_model_alias(
            name=cfg.model_name, alias=cfg.target_alias, version=str(version)
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Could not set alias '%s': %s", cfg.target_alias, exc)

# ----------------------------- core -----------------------------

def run_template(cfg: Config) -> tuple[str, int, float, float]:
    """Execute end-to-end training and registration. Returns (model_uri, version, train_ms, predict_ms)."""
    spark = SparkSession.builder.getOrCreate()

    # MLflow setup
    mlflow.set_experiment(cfg.experiment_name)
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tag("release.stage", cfg.env)
    mlflow.set_tag("run.profile", cfg.env)

    # Get Training data
    training_data_impl = TrainingDataConfig()
    
    _training_data_call_args = dict(cfg.training_data_config)
    _training_data_call_args[CONFIG_DEFAULT_CATALOG_NAME] = cfg.catalog_name
    _training_data_call_args[CONFIG_ENV] = cfg.env

    x_pdf, y_pdf = training_data_impl.get_training_data(_training_data_call_args)

    # Prepare Data for Train
    maybe_features = cfg.training_data_config.get(CONFIG_FEATURE_COLUMNS)
    if maybe_features and cfg.model_config:
        cfg.model_config[CONFIG_FEATURE_COLUMNS] = maybe_features
    maybe_temporal_column_from_training = cfg.training_data_config.get(CONFIG_TEMPORAL_COLUMN_NAME)
    if maybe_temporal_column_from_training and cfg.model_config:
        cfg.model_config[CONFIG_TEMPORAL_COLUMN_NAME] = maybe_temporal_column_from_training

    # Split
    train_test_split_impl = SplitContractConfig()
    if maybe_features:

        final_split_selection = maybe_features

        if cfg.split_config.get(CONFIG_SPLIT_STRATEGY) == SPLIT_TIME:
            LOG.info(f"'{CONFIG_SPLIT_STRATEGY}' set to '{SPLIT_TIME}': verifying training data config params")

            maybe_temporal_column_from_split = cfg.split_config.get(CONFIG_TEMPORAL_COLUMN_NAME)
            if maybe_temporal_column_from_training and maybe_temporal_column_from_split:
                if maybe_temporal_column_from_training != maybe_temporal_column_from_split:
                    raise ValueError(f"Different {CONFIG_TEMPORAL_COLUMN_NAME} defined in training and split.")
            final_temporal_column = maybe_temporal_column_from_training or maybe_temporal_column_from_split
            if final_temporal_column:
                LOG.info(f"'{CONFIG_TEMPORAL_COLUMN_NAME}' set: '{final_temporal_column}'.")
                cfg.model_config[CONFIG_TEMPORAL_COLUMN_NAME] = final_temporal_column
                if final_temporal_column not in maybe_features:
                    LOG.info(f"Column '{final_temporal_column}' not on features, adding to selection")
                    final_split_selection.append(final_temporal_column)
                    maybe_discarded = cfg.model_config.get(CONFIG_SECTION_DISCARDED_FEATURES, [])
                    cfg.model_config[CONFIG_SECTION_DISCARDED_FEATURES] = list(set(maybe_discarded + [final_temporal_column]))

        x_pdf = x_pdf[final_split_selection]

    all_split_set = train_test_split_impl.split(x_pdf, y_pdf, cfg.split_config)

    # Train Model
    model_impl = _load_model_impl(cfg)
    maybe_random_state_from_split = cfg.split_config.get(CONFIG_RANDOM_STATE)
    maybe_random_state_from_model = cfg.model_config.get(CONFIG_RANDOM_STATE)
    if maybe_random_state_from_split and not maybe_random_state_from_model:
        cfg.model_config[CONFIG_RANDOM_STATE] = maybe_random_state_from_split

    _model_call_args = dict(cfg.model_config)
    _model_call_args[CONFIG_DEFAULT_CATALOG_NAME] = cfg.catalog_name
    _model_call_args[CONFIG_ENV] = cfg.env

    model = model_impl.get_model(_model_call_args)
    model = _maybe_wrap_model_for_prediction_method(cfg, model)
    model, train_ms = _time_fit(
        model=model,
        X=all_split_set[X_TRAIN],
        y=all_split_set[Y_TRAIN],
        metadata=_get_metadata(cfg, all_split_set),
    )
    raw_preds, predict_ms = _time_predict(model, all_split_set[X_TEST])
    preds, prediction_proba = _split_predict_output(raw_preds)

    # Signature
    signature, input_example = _infer_signature(model, all_split_set[X_TRAIN])

    # Log & register
    mlflow.end_run()
    run_id = _register_model(cfg, input_example, model, model_impl, predict_ms, signature, train_ms)

    _register_model_card(cfg)

    version = _latest_model_version(cfg.model_name)
    model_uri = f"models:/{cfg.model_name}/{version}"

    _register_model_alias(cfg, version)

    _tag_framework_version(cfg)

    _set_task_values(model_uri, cfg.model_name, version)

    _append_latency_metrics(spark, cfg, run_id, version, train_ms, predict_ms)

    _create_baseline_table(all_split_set, cfg, preds, prediction_proba, spark, version)

    return model_uri, version, train_ms, predict_ms

# ----------------------------- cli -----------------------------

def _parse_args(argv: list[str]) -> Config:
    ap = argparse.ArgumentParser(description="Framework training")
    ap.add_argument("--env", default="dev")
    ap.add_argument("--catalog_name", required=True)
    ap.add_argument("--experiment_name", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--databricks_mlops_stack_version", default="")
    ap.add_argument("--training_data_config")
    ap.add_argument("--model_config")
    ap.add_argument("--model_tuning_config")
    ap.add_argument("--model_card_path", default="")
    ap.add_argument("--metrics_latency_table", default="")
    ap.add_argument("--split_config", default="", )
    ap.add_argument("--quality_monitor_config", default="", )
    ap.add_argument("--validation_config", default="")
    ap.add_argument("--baseline_table_name", required=True, help="<catalog>.<schema>.<table>_baseline")

    args = ap.parse_args(argv)

    split_config_str = (args.split_config or "").strip()
    model_config_str = (args.model_config or "").strip()
    model_tuning_config_str = (args.model_tuning_config or "").strip()
    training_data_config_str = (args.training_data_config or "").strip()
    quality_monitor_config_str = (args.quality_monitor_config or "").strip()
    model_config=YamlUtils.yaml_to_dict(model_config_str) if model_config_str else {}
    training_data_config = YamlUtils.yaml_to_dict(training_data_config_str) if training_data_config_str else {}
    validation_config_str = (args.validation_config or "").strip()

    if model_tuning_config_str:
        logging.info("Tuning config detected, merging configs")
        model_tuning_config = YamlUtils.yaml_to_dict(model_tuning_config_str)
        model_config = model_tuning_config | model_config


    config = Config(env=args.env,
                    catalog_name=args.catalog_name,
                    experiment_name=args.experiment_name,
                    model_name=args.model_name,
                    databricks_mlops_stack_version=args.databricks_mlops_stack_version,
                    training_data_config=training_data_config,
                    model_config=model_config,
                    model_card_path=args.model_card_path,
                    split_config=YamlUtils.yaml_to_dict(split_config_str) if split_config_str else {},
                    metrics_latency_table=args.metrics_latency_table.strip(),
                    quality_monitor_config=YamlUtils.yaml_to_dict(quality_monitor_config_str) if quality_monitor_config_str else {},
                    validation_config=YamlUtils.yaml_to_dict(validation_config_str) if validation_config_str else {},
                    baseline_table_name=args.baseline_table_name)

    print("Configurations:")
    print(f"=== Model config:{config.model_config}")
    print(f"=== Training config:{config.training_data_config}")
    print(f"=== Split config:{config.split_config}")
    print(f"=== Quality monitor config:{config.quality_monitor_config}")

    return config


def main(argv: list[str] | None = None) -> int:
    LOG.info("Running Training.....")
    cfg = _parse_args(argv or sys.argv[1:])
    model_uri, version, train_ms, predict_ms = run_template(cfg)
    print("=== Training Complete ===")
    print(f"Model URI: {model_uri}")
    print(f"Model Version: {version}")
    print(f"Train Time (ms): {train_ms:.2f}")
    print(f"Predict Time (ms): {predict_ms:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
