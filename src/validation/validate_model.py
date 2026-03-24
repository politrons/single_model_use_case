""""
Databricks Framework: Validation/Evaluation job (CLI)

Refactors the original validation notebook into a Python script with `main()` so it can
be executed directly from bundle source using `spark_python_task`.

- Comments are in English
- Strong, explicit structure (dataclass Config, small pure helpers)
- Uses structural pattern matching for env tagging
- Pulls model name/version from a dependency task via dbutils when provided
- Evaluates with `mlflow.models.evaluate`, logs artifact of metrics, writes tall metrics table

Skip behavior:
- If validation_config.skip_evaluation is truthy (true/"true"/1/"1"/yes/on), the evaluation block is skipped.
- If skip_evaluation is missing or falsy, evaluation runs as usual.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlflow  # type: ignore # noqa
import pandas as pd  # type: ignore # noqa
from mlflow import MlflowClient  # type: ignore # noqa
from mlflow.models import EvaluationResult  # type: ignore # noqa
from pyspark.sql import Row, SparkSession  # type: ignore # noqa

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

from databricks_mlops_stack.training.model.predict_and_proba_wrapper import (  # type: ignore # noqa
    PredictAndProbaWrapper,
)
from databricks_mlops_stack.training.data.training_data_config import TrainingDataConfig  # type: ignore # noqa
from databricks_mlops_stack.split.split_config import SplitContractConfig  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.core import (  # type: ignore # noqa
    CONFIG_ENV,
    CONFIG_DEFAULT_CATALOG_NAME,
    CONFIG_TARGET_COLUMN,
)
from databricks_mlops_stack.utils.mlops_utils import YamlUtils  # type: ignore # noqa
from databricks_mlops_stack.validation.eval_model.eval_model_config import EvalModelConfig  # type: ignore # noqa
from databricks_mlops_stack.validation.eval_result.eval_result_config import EvalResultConfig  # type: ignore # noqa

# ----------------------------- logging -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)
LOG = logging.getLogger("framework.validate")


# ----------------------------- config -----------------------------
@dataclass(frozen=True)
class Config:
    env: str
    catalog_name: str
    experiment_name: str
    model_name: str
    model_version: str | None
    dependency_task_key: str | None
    training_data_config: dict[str, Any]
    split_config: dict[str, Any]
    validation_config: dict[str, Any]
    model_config: dict[str, Any]
    eval_result_config: dict[str, dict[str, float | str]]
    metrics_table: str | None


# ----------------------------- helpers -----------------------------

def _parse_boolish(value: Any) -> bool | None:
    """Parse common boolean-like inputs.

    Returns:
        True/False when value looks like a boolean, otherwise None.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off"}:
            return False

    return None


def _should_skip_evaluation(cfg: Config) -> bool:
    """Return True only if validation_config.skip_evaluation is explicitly truthy.

    If the key does not exist (or is unparseable), we default to evaluating.
    """
    raw = None
    try:
        raw = (cfg.validation_config or {}).get("skip_evaluation")
    except Exception:
        raw = None

    parsed = _parse_boolish(raw)
    return parsed is True


def _load_eval_model_impl() -> EvalModelConfig:
    LOG.info("Loading eval model from config....")
    return EvalModelConfig()


def _load_eval_result_impl() -> EvalResultConfig:
    LOG.info("Loading eval result from config....")
    return EvalResultConfig()


def _ensure_targets(eval_df: pd.DataFrame, y: pd.Series | pd.DataFrame, targets: str) -> pd.DataFrame:
    """Attach targets column(s) to eval_df. Supports comma-separated list."""
    cols = [t.strip() for t in targets.split(",") if t.strip()]
    if not cols:
        raise ValueError("No targets provided")

    if isinstance(y, pd.Series):
        if len(cols) != 1:
            raise ValueError("Received a single target series but multiple target names were provided")
        eval_df[cols[0]] = y
    else:  # DataFrame
        for c in cols:
            if c not in y.columns:
                raise ValueError(f"Target column '{c}' not found in y DataFrame")
            eval_df[c] = y[c]
    return eval_df


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


def _resolve_model_identity(cfg: Config) -> tuple[str, str]:
    """Return (model_name, model_version_str) considering dependency_task_key and fallbacks."""
    model_name = cfg.model_name or ""
    model_version = (cfg.model_version or "").strip()

    # Try to pull from upstream task values if requested
    if cfg.dependency_task_key:
        try:
            from mlflow.utils.databricks_utils import dbutils  # type: ignore

            if not model_name:
                model_name = dbutils.jobs.taskValues.get(cfg.dependency_task_key, "model_name", debugValue="")
            if not model_version:
                model_version = dbutils.jobs.taskValues.get(cfg.dependency_task_key, "model_version", debugValue="")
        except Exception as exc:
            LOG.info("dbutils.jobs.taskValues.get failed or unavailable: %s", exc)

    if not model_name:
        raise ValueError("model_name is required (either as argument or via dependency_task_key)")

    if not model_version:
        # Fallback to latest
        model_version = str(_latest_model_version(model_name))
        LOG.info("No model_version provided; resolved latest version=%s for model '%s'", model_version, model_name)

    return model_name, model_version


def _log_metrics_artifact(model_name: str, eval_result: EvaluationResult) -> None:
    """Log evaluation metrics as a text artifact without writing to local FS."""
    lines: list[str] = []
    header = f"{'metric_name':30}  {'candidate':30}"
    lines.append(header)

    for metric, val in eval_result.metrics.items():
        line = f"{metric:30}  {str(val):30}"
        lines.append(line)

    metrics_text = "\n".join(lines)

    mlflow.log_text(
        metrics_text,
        artifact_file=f"eval/{model_name}_metrics.txt",
    )


def _append_metrics_table(
    spark: SparkSession,
    table: str,
    experiment_name: str,
    run_id: str,
    model_name: str,
    model_version: int,
    model_type: str,
    eval_result: EvaluationResult,
    X_test: pd.DataFrame,
) -> None:
    ts_rows: list[Row] = []
    from datetime import datetime, timezone

    base = dict(
        ts=datetime.now(timezone.utc),
        experiment_name=experiment_name,
        run_id=run_id,
        model_name=model_name,
        model_version=int(model_version) if str(model_version).strip() else None,
        model_type=model_type,
    )

    def _safe_float(x: Any) -> float | None:
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except Exception:
            return None

    for mname, mval in eval_result.metrics.items():
        ts_rows.append(Row(**base, metric_name=mname, metric_value=_safe_float(mval)))

    try:
        ts_rows.append(Row(**base, metric_name="val_rows", metric_value=float(len(X_test))))
        ts_rows.append(Row(**base, metric_name="val_features", metric_value=float(X_test.shape[1])))
    except Exception:
        pass

    (
        spark.createDataFrame(ts_rows)
        .write.mode("append")
        .format("delta")
        .option("mergeSchema", "true")
        .saveAsTable(table)
    )
    LOG.info("Appended %d rows to %s", len(ts_rows), table)


def _normalized_targets_arg(targets: Any) -> Any:
    return targets if "," not in str(targets) else [t.strip() for t in str(targets).split(",") if t.strip()]


def _resolve_model_for_evaluate(model_uri: str) -> Any:
    try:
        loaded_model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        LOG.info(
            "Could not load sklearn model from URI '%s'. Using URI directly in evaluate(). Error: %s",
            model_uri,
            exc,
        )
        return model_uri

    internal_model = None
    internal_model_key = None
    if isinstance(loaded_model, PredictAndProbaWrapper):
        internal_model = loaded_model.internal_model
        internal_model_key = loaded_model.internal_model_key

    if internal_model is None:
        LOG.info(
            "Loaded model '%s' is not a wrapper (type=%s). Using model URI in evaluate().",
            model_uri,
            type(loaded_model).__name__,
        )
        return model_uri

    LOG.info(
        "PredictAndProbaWrapper detected for '%s'. Logging internal model '%s' as sklearn "
        "so mlflow.evaluate() can access predict_proba for probability-based metrics.",
        model_uri,
        internal_model_key,
    )

    artifact_path = "eval_internal_model"
    info = mlflow.sklearn.log_model(internal_model, artifact_path)
    internal_model_uri = info.model_uri
    LOG.info("Internal model logged at '%s'. Loading as pyfunc for evaluate().", internal_model_uri)
    return mlflow.pyfunc.load_model(internal_model_uri)


def _evaluate_model(cfg: Config, eval_df: pd.DataFrame, model_uri: str) -> EvaluationResult| None:
    targets = cfg.validation_config.get("targets")
    targets_arg = _normalized_targets_arg(targets)
    model_for_evaluate = _resolve_model_for_evaluate(model_uri)
    eval_model_impl = _load_eval_model_impl()
    evaluator_config = {
        "default": {
            "log_model_explainability": False,
        }
    }
    try:
        eval_result = eval_model_impl.evaluate(
            model_for_evaluate=model_for_evaluate,
            data=eval_df,
            targets=targets_arg,
            model_type=cfg.validation_config.get("model_type"),
            evaluators=["default"],
            evaluator_config=evaluator_config,
        )
    except Exception as exc:
        LOG.error( "Could not evaluate model '%s': %s", model_uri, exc,)
        return None
    return eval_result


def infer_validation_config_from_external_configs(cfg: Config) -> None:
    if cfg.training_data_config and cfg.training_data_config.get(CONFIG_TARGET_COLUMN):
        target = cfg.training_data_config[CONFIG_TARGET_COLUMN]
        LOG.info("Replacing target '%s' on validation config", target)
        cfg.validation_config["targets"] = target


# ----------------------------- core -----------------------------

def run_template(cfg: Config) -> tuple[str, str]:
    spark = SparkSession.builder.getOrCreate()

    # MLflow setup
    mlflow.set_experiment(cfg.experiment_name)
    mlflow.set_registry_uri("databricks-uc")

    # Resolve model identity
    model_name, model_version = _resolve_model_identity(cfg)
    model_uri = f"models:/{model_name}/{model_version}"
    LOG.info("Evaluating model URI: %s", model_uri)

    # Get training data
    training_data_impl = TrainingDataConfig()

    _training_data_call_args = dict(cfg.training_data_config)
    _training_data_call_args[CONFIG_DEFAULT_CATALOG_NAME] = cfg.catalog_name
    _training_data_call_args[CONFIG_ENV] = cfg.env
    
    x_pdf, y_pdf = training_data_impl.get_training_data(_training_data_call_args)

    # Split/transform
    train_test_split_impl = SplitContractConfig()
    all_split_set = train_test_split_impl.split(x_pdf, y_pdf, cfg.split_config)

    infer_validation_config_from_external_configs(cfg)

    X_test = all_split_set["X_test"]
    eval_df = X_test.copy()
    targets: str = cast(str, cfg.validation_config["targets"])
    eval_df = _ensure_targets(eval_df, all_split_set["y_test"], targets)

    # Load eval result
    eval_result_impl = _load_eval_result_impl()

    # --- Skip evaluation if configured ---
    if _should_skip_evaluation(cfg):
        LOG.info("skip_evaluation=true -> Skipping evaluation block (no mlflow.start_run, no evaluate, no alias).")
        return model_uri, str(model_version)

    # Evaluate
    mlflow.end_run()
    with mlflow.start_run(run_name=f"evaluate_{cfg.env}") as run:
        run_id = run.info.run_id
        eval_result: EvaluationResult | None = _evaluate_model(cfg, eval_df, model_uri)
        eval_result_impl.eval_result(eval_result, cfg.eval_result_config)

        if eval_result is not None:
            # Plain-text metrics artifact
            _log_metrics_artifact(cfg.model_name, eval_result)

        LOG.info("Validation checks passed. Assigning 'challenger' alias to version %s.", model_version)
        client = MlflowClient(registry_uri="databricks-uc")
        client.set_registered_model_alias(model_name, "challenger", str(model_version))

        # Persist tall metrics table if requested
        if cfg.metrics_table and eval_result is not None:
            model_type: str = cast(str, cfg.validation_config["model_type"])
            _append_metrics_table(
                spark=spark,
                table=cfg.metrics_table,
                experiment_name=cfg.experiment_name,
                run_id=run_id,
                model_name=model_name,
                model_version=int(model_version),
                model_type=model_type,
                eval_result=eval_result,
                X_test=X_test,
            )
        elif cfg.metrics_table and eval_result is None:
            LOG.info("Skipping metrics table append because evaluation result is None.")

    return model_uri, str(model_version)


# ----------------------------- cli -----------------------------

def _parse_args(argv: list[str]) -> Config:
    ap = argparse.ArgumentParser(description="Framework evaluation job (minimal)")
    ap.add_argument("--env", default="dev")
    ap.add_argument("--experiment_name", required=True)
    ap.add_argument("--catalog_name", required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--model_version", default="")
    ap.add_argument("--dependency_task_key", default="")
    ap.add_argument("--training_data_config")
    ap.add_argument("--model_config", default="")
    ap.add_argument("--split_config", default="")
    ap.add_argument("--validation_config", default="")
    ap.add_argument("--eval_result_config")
    ap.add_argument("--metrics_table", default="")
    args = ap.parse_args(argv)

    training_data_config_str = (args.training_data_config or "").strip()
    split_config_str = (args.split_config or "").strip()
    validation_config_str = (args.validation_config or "").strip()
    model_config_str = (args.model_config or "").strip()
    eval_result_config_str = (args.eval_result_config or "").strip()
    training_data_config = YamlUtils.yaml_to_dict(training_data_config_str) if training_data_config_str else {}
    model_config = YamlUtils.yaml_to_dict(model_config_str) if model_config_str else {}

    config = Config(
        env=args.env,
        catalog_name=args.catalog_name,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        model_version=(args.model_version or None),
        dependency_task_key=(args.dependency_task_key or None),
        training_data_config=training_data_config,
        model_config=model_config,
        split_config=YamlUtils.yaml_to_dict(split_config_str) if split_config_str else {},
        validation_config=YamlUtils.yaml_to_dict(validation_config_str) if validation_config_str else {},
        eval_result_config=YamlUtils.yaml_to_dict(eval_result_config_str) if eval_result_config_str else {},
        metrics_table=(args.metrics_table.strip() or None),
    )

    print("Configurations:")
    print(f"=== Model config:{config.model_config}")
    print(f"=== Training config:{config.training_data_config}")
    print(f"=== Validation config:{config.validation_config}")
    print(f"=== Split config:{config.split_config}")
    print(f"=== Evaluation config:{config.eval_result_config}")

    return config


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(argv or sys.argv[1:])
    model_uri, version = run_template(cfg)
    print("\n=== Validation Complete ===")
    print(f"Model URI: {model_uri}")
    print(f"Model Version: {version}")
    return 0


if __name__ == "__main__":
    main()
