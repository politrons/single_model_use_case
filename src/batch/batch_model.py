from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow  # type: ignore # noqa
import pandas as pd  # type: ignore # noqa
from mlflow import MlflowClient  # type: ignore # noqa
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

from databricks_mlops_stack.split.split_config import SplitContractConfig  # type: ignore # noqa
from databricks_mlops_stack.training.data.training_data_config import TrainingDataConfig  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.core import (  # type: ignore # noqa
    CONFIG_DEFAULT_CATALOG_NAME,
    CONFIG_ENV,
)
from databricks_mlops_stack.utils.mlops_utils import YamlUtils  # type: ignore # noqa

# ----------------------------- logging -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
LOG = logging.getLogger("framework.batch")


@dataclass(frozen=True)
class Config:
    env: str
    catalog_name: str
    model_name: str
    output_table: str
    training_data_config: dict[str, Any]
    split_config: dict[str, Any]
    experiment_name: str = ""
    model_uri: str = ""
    model_alias: str = "champion"


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


def _resolve_model_uri(cfg: Config) -> str:
    direct_uri = (cfg.model_uri or "").strip()
    if direct_uri:
        return direct_uri

    alias = (cfg.model_alias or "").strip()
    if alias:
        try:
            mv = MlflowClient().get_model_version_by_alias(cfg.model_name, alias)
            return f"models:/{cfg.model_name}/{mv.version}"
        except Exception:
            LOG.info("Alias '%s' not found for model '%s'. Falling back to latest version.", alias, cfg.model_name)

    latest = _latest_model_version(cfg.model_name)
    if latest <= 0:
        raise ValueError(f"No model versions found for '{cfg.model_name}'")
    return f"models:/{cfg.model_name}/{latest}"


def _to_prediction_series(raw: Any, length: int) -> pd.Series:
    if isinstance(raw, pd.Series):
        return raw.reset_index(drop=True)
    if isinstance(raw, pd.DataFrame):
        if "prediction" in raw.columns:
            return raw["prediction"].reset_index(drop=True)
        return raw.iloc[:, 0].reset_index(drop=True)
    if isinstance(raw, (list, tuple)):
        return pd.Series(list(raw)[:length])
    return pd.Series([raw] * length)


def _probability_to_json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        maybe_na = pd.isna(value)
        if isinstance(maybe_na, bool) and maybe_na:
            return None
    except Exception:
        pass

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.startswith("{") or s.startswith("["):
            return s
        try:
            return json.dumps({"0": float(s)})
        except Exception:
            return json.dumps({"0": s})

    if isinstance(value, dict):
        try:
            return json.dumps({str(k): float(v) for k, v in value.items()})
        except Exception:
            return json.dumps({str(k): str(v) for k, v in value.items()})

    if isinstance(value, (list, tuple)):
        try:
            return json.dumps({str(i): float(v) for i, v in enumerate(value)})
        except Exception:
            return json.dumps({str(i): str(v) for i, v in enumerate(value)})

    try:
        return json.dumps({"0": float(value)})
    except Exception:
        return json.dumps({"0": str(value)})


def _to_probability_series(raw: Any, length: int) -> pd.Series:
    if isinstance(raw, pd.Series):
        values = raw.reset_index(drop=True).tolist()
    elif isinstance(raw, pd.DataFrame):
        values = raw.iloc[:, 0].reset_index(drop=True).tolist()
    elif isinstance(raw, (list, tuple)):
        values = list(raw)[:length]
    else:
        values = [raw] * length
    return pd.Series([_probability_to_json(v) for v in values])


def _split_predict_output(raw_output: Any) -> tuple[Any, Any | None]:
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
        preds = raw_output[pred_col] if pred_col is not None else raw_output
        probs = raw_output[prob_col] if prob_col is not None else None
        return preds, probs

    if isinstance(raw_output, dict):
        preds = raw_output.get("prediction", raw_output.get("predictions"))
        probs = raw_output.get(
            "prediction_proba",
            raw_output.get("probability", raw_output.get("probabilities", raw_output.get("proba"))),
        )
        if preds is not None:
            return preds, probs

    if isinstance(raw_output, tuple) and len(raw_output) == 2:
        return raw_output[0], raw_output[1]

    return raw_output, None


def _build_predictions_df(
    x_df: pd.DataFrame,
    y_df: pd.Series | pd.DataFrame | None,
    split_name: str,
    model: Any,
    model_uri: str,
    cfg: Config,
) -> pd.DataFrame:
    raw_preds = model.predict(x_df)
    preds, probs = _split_predict_output(raw_preds)

    out = x_df.copy()
    out["prediction"] = _to_prediction_series(preds, len(out))
    if probs is not None:
        out["prediction_proba"] = _to_probability_series(probs, len(out))

    if isinstance(y_df, pd.Series):
        out["target"] = y_df.reset_index(drop=True)
    elif isinstance(y_df, pd.DataFrame) and len(y_df.columns) > 0:
        for c in y_df.columns:
            out[f"target_{c}"] = y_df[c].reset_index(drop=True)

    out["split_name"] = split_name
    out["model_name"] = cfg.model_name
    out["model_uri"] = model_uri
    out["env"] = cfg.env
    out["prediction_ts"] = datetime.now(timezone.utc).isoformat()
    return out


def run_template(cfg: Config) -> tuple[str, int]:
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")
    if cfg.experiment_name:
        mlflow.set_experiment(cfg.experiment_name)

    model_uri = _resolve_model_uri(cfg)
    model = mlflow.pyfunc.load_model(model_uri)

    training_data_impl = TrainingDataConfig()
    call_args = dict(cfg.training_data_config)
    call_args[CONFIG_DEFAULT_CATALOG_NAME] = cfg.catalog_name
    call_args[CONFIG_ENV] = cfg.env
    x_pdf, y_pdf = training_data_impl.get_training_data(call_args)

    split_impl = SplitContractConfig()
    all_split_set = split_impl.split(x_pdf, y_pdf, cfg.split_config)

    output_frames: list[pd.DataFrame] = []
    for k, x_split in all_split_set.items():
        if not str(k).startswith("X_"):
            continue
        if not isinstance(x_split, pd.DataFrame):
            continue
        y_key = f"y_{str(k)[2:]}"
        y_split = all_split_set.get(y_key)
        output_frames.append(_build_predictions_df(x_split, y_split, k, model, model_uri, cfg))

    if not output_frames:
        raise ValueError("No split DataFrames found to predict (expected keys starting with 'X_').")

    final_pdf = pd.concat(output_frames, ignore_index=True)
    (
        spark.createDataFrame(final_pdf)
        .write
        .mode("append")
        .format("delta")
        .option("mergeSchema", "true")
        .saveAsTable(cfg.output_table)
    )
    LOG.info("Wrote %d rows into %s", len(final_pdf), cfg.output_table)
    return model_uri, len(final_pdf)


def _parse_args(argv: list[str]) -> Config:
    ap = argparse.ArgumentParser(description="Batch scoring job")
    ap.add_argument("--env", default="dev")
    ap.add_argument("--catalog_name", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--output_table", required=True)
    ap.add_argument("--experiment_name", default="")
    ap.add_argument("--model_uri", default="")
    ap.add_argument("--model_alias", default="champion")
    ap.add_argument("--training_data_config", required=True)
    ap.add_argument("--split_config", required=True)
    args = ap.parse_args(argv)

    return Config(
        env=args.env,
        catalog_name=args.catalog_name,
        model_name=args.model_name,
        output_table=args.output_table,
        experiment_name=args.experiment_name,
        model_uri=args.model_uri,
        model_alias=args.model_alias,
        training_data_config=YamlUtils.yaml_to_dict((args.training_data_config or "").strip()),
        split_config=YamlUtils.yaml_to_dict((args.split_config or "").strip()),
    )


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(argv or sys.argv[1:])
    model_uri, rows = run_template(cfg)
    print("\n=== Batch Complete ===")
    print(f"Model URI: {model_uri}")
    print(f"Output Table: {cfg.output_table}")
    print(f"Rows Written: {rows}")
    return 0


if __name__ == "__main__":
    main()

