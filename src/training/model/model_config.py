from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow  # type: ignore # noqa
import databricks_mlops_stack  # type: ignore # noqa

from databricks_mlops_stack.training.model.hyperparam_searches.base_search import hyperparam_search  # type: ignore # noqa
from databricks_mlops_stack.training.model.transformers.features_transformer import build_features_preprocessing  # type: ignore # noqa
from databricks_mlops_stack.training.model.transformers.target_transformer import build_target_transformer  # type: ignore # noqa
from databricks_mlops_stack.training.model.samplers.base_samplers import build_sampler_preprocessing  # type: ignore # noqa
from databricks_mlops_stack.training.model.algorithms.base_algos import build_configured_estimator  # type: ignore # noqa
from databricks_mlops_stack.training.model.pipeline.base_pipe import build_pipeline_model  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.core import (  # type: ignore # noqa
    CONFIG_RANDOM_STATE,
)
from databricks_mlops_stack.utils.constants.model import (  # type: ignore # noqa
    CONFIG_SECTION_EARLY_STOPPING,
    CONFIG_SECTION_MODEL,
    CONFIG_MODEL_CLASS,
    ALGO_NEURALPROPHET,
)

def _model_has_early_stop(config: dict[str, Any]) -> bool:
    return config.get(CONFIG_SECTION_EARLY_STOPPING) is not None

def _model_is_neuralprophet(config: dict[str, Any]) -> bool:
    model_config = config.get(CONFIG_SECTION_MODEL)
    if not model_config:
        return False
    model_base_class = model_config.get(CONFIG_MODEL_CLASS)
    if not model_base_class:
        return False
    if model_base_class.lower().strip() == ALGO_NEURALPROPHET:
        return True
    else:
        return False

class ModelContractConfig:

    def get_model(self, config: dict[str, Any]):
        """Build a model pipeline or a search object according to ``config``.

        Expected keys in ``config``
        ---------------------------
        - ``features``: list[str] — all feature columns.
        - ``discarded_features``: list[str] — features to drop early.
        - ``features_transformers``: list[dict] — optional, see :func:`build_preprocessing`.
        - ``random_state``: int — seed for reproducibility.
        - ``model``: {``type``, ``task``, ``params``}
            - ``type`` ∈ {"scikit.random_forest", "xgboost", "lightgbm"}
            - ``task`` ∈ {"classification", "regression"}
            - ``params``: estimator kwargs.
        - ``sampling`` (optional): {``type``: "SMOTE", ``params``: {...}}
        - ``target_transformer`` (optional, regression only): {``type``, ``params``}
        - ``hyperparam_search`` (optional):
            - ``search_strategy``: "grid" (others future)
            - ``model_params_space``: dict[str, list]
            - ``num_folds``: int (default 3)
            - ``split_strategy``: "general" | "time_series"

        Returns
        -------
        sklearn.base.BaseEstimator
            Either a ready-to-fit pipeline, or a configured scikit-learn ``SearchCV`` instance.
        """
        random_state = config[CONFIG_RANDOM_STATE]
        has_early_stop_config = _model_has_early_stop(config)
        is_neuralprophet = _model_is_neuralprophet(config)

        col_transform = build_features_preprocessing(config, is_neuralprophet)

        maybe_sampler = build_sampler_preprocessing(config)

        estimator = build_configured_estimator(config, random_state)

        pipe = build_pipeline_model(col_transform, maybe_sampler, has_early_stop_config, estimator)

        pipe = build_target_transformer(config, pipe)

        pipe = hyperparam_search(config, pipe, random_state)

        return pipe

    def log_model(self, model, name, signature, input_example, args: dict[str, Any]):
        package_dir = Path(databricks_mlops_stack.__file__).resolve().parent
        code_paths = [str(package_dir)]

        # Build a temporary local model first and reuse the exact requirements inferred by MLflow there.
        # Then remove only the private framework package that serving cannot install from public indexes.
        default_reqs, inferred = self.get_infer_mlflow_dependencies(code_paths, input_example, model, signature)

        filtered = [r for r in inferred if not str(r).lower().startswith("databricks-mlops-stack")]
        # Preserve order while removing duplicates.
        filtered = list(dict.fromkeys(filtered))
        if not filtered:
            filtered = default_reqs

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=name,
            signature=signature,
            input_example=input_example,
            code_paths=code_paths,
            pip_requirements=filtered,
        )

    def get_infer_mlflow_dependencies(self, code_paths: list[str], input_example, model, signature) -> tuple[
        list[str], list[str]]:
        default_reqs = mlflow.sklearn.get_default_pip_requirements()
        try:
            with TemporaryDirectory(prefix="mlflow-model-reqs-") as tmp_dir:
                local_model_path = str(Path(tmp_dir) / "model")
                mlflow.sklearn.save_model(
                    sk_model=model,
                    path=local_model_path,
                    signature=signature,
                    input_example=input_example,
                    code_paths=code_paths,
                )
                inferred: list[str] = []
                req_file = Path(local_model_path) / "requirements.txt"
                if req_file.exists():
                    for line in req_file.read_text(encoding="utf-8").splitlines():
                        dep = line.strip()
                        if not dep or dep.startswith("#"):
                            continue
                        inferred.append(dep)
                if not inferred:
                    inferred = list(
                        mlflow.models.infer_pip_requirements(
                            model_uri=local_model_path,
                            flavor="sklearn",
                            fallback=default_reqs,
                        )
                    )
        except Exception:
            inferred = default_reqs
        return default_reqs, inferred


build = ModelContractConfig()
