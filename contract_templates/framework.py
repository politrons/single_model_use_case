from typing import Any
import logging
from importlib.metadata import PackageNotFoundError, version

import mlflow
from databricks_mlops_stack.training.model.model_contract import ModelContract  # type: ignore # noqa

import databricks_mlops_stack  # type: ignore # noqa
from pathlib import Path  # type: ignore # noqa
from tempfile import TemporaryDirectory  # type: ignore # noqa
import gc  # type: ignore # noqa

logger = logging.getLogger(__name__)

# ── Model configuration ──────────────────────────────────────────── #

_NUMERICAL_FEATURES: list[str] = {{NUMERICAL_FEATURES}}

_SEGMENT_COLUMNS: list[str] = {{SEGMENT_COLUMNS}}

_CONFIG: dict = {{CONFIG}}

_RANDOM_STATE: int = {{RANDOM_STATE}}

_BASE_PARAMS: dict = {{BASE_PARAMS}}

_EXTRA_PARAMS: dict = {{EXTRA_PARAMS}}

_N_JOBS: int = {{N_JOBS}}

_IS_TENSORFLOW: bool = {{IS_TENSORFLOW}}

_FACTORY_FN = {{FACTORY_FN_NAME}}


# ── Contract implementation ──────────────────────────────────────── #

class ModelContractImpl(ModelContract):

    @staticmethod
    def _merge_requirements(primary: list[str], fallback: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for req in [*primary, *fallback]:
            text = str(req).strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            merged.append(text)
        return merged

    @staticmethod
    def _resolve_requirement(distribution_name: str) -> str:
        try:
            return f"{distribution_name}=={version(distribution_name)}"
        except PackageNotFoundError:
            return distribution_name
        except Exception:
            return distribution_name

    def _minimum_runtime_requirements(self) -> list[str]:
        """
        Return a safety baseline of runtime deps to avoid empty/default-only artifacts.

        MLflow dependency inference can silently fallback (especially for large models).
        This baseline keeps serving environments usable even when inference degrades.
        """
        required = [
            "mlflow",
            "cloudpickle",
            "scikit-learn",
            "numpy",
            "pandas",
            "scipy",
            "psutil",
        ]

        if _IS_TENSORFLOW or "tf" in globals():
            required.append("tensorflow")
        if "Prophet" in globals():
            required.extend(["prophet", "python-dateutil"])

        return [self._resolve_requirement(pkg) for pkg in required]

    def get_model(self, args: dict[str, Any]):
        model = MultiClusterWrapper(
            numerical_features=_NUMERICAL_FEATURES,
            segment_columns=_SEGMENT_COLUMNS,
            model_factory=_FACTORY_FN,
            is_tensorflow=_IS_TENSORFLOW,
            config=_CONFIG,
            random_state=_RANDOM_STATE,
            base_params=_BASE_PARAMS,
            extra_params=_EXTRA_PARAMS,
            n_jobs=_N_JOBS,
        )
        return model

    def log_model(
        self,
        model,
        model_name,
        signature,
        input_example,
        args: dict[str, Any],
    ):
        if hasattr(model, "compact_for_serialization"):
            try:
                model.compact_for_serialization()
            except Exception:
                pass
        gc.collect()
        package_dir = Path(databricks_mlops_stack.__file__).resolve().parent
        code_paths = [str(package_dir)]

        # Build a temporary local model first and reuse the exact requirements inferred by MLflow there.
        # Then remove only the private framework package that serving cannot install from public indexes.
        logger.info("Starting dependency inference for model '%s'", model_name)
        default_reqs, inferred = self.get_infer_mlflow_dependencies(code_paths, input_example, model, signature)
        logger.info(
            "Dependency inference result for model '%s': default=%d inferred=%d",
            model_name,
            len(default_reqs),
            len(inferred),
        )

        minimum_reqs = self._minimum_runtime_requirements()
        if set(map(str.lower, inferred)) == set(map(str.lower, default_reqs)):
            logger.warning(
                "Model '%s' produced default-only inferred requirements. "
                "Adding minimum runtime requirements to avoid under-specified serving env.",
                model_name,
            )

        merged = self._merge_requirements(inferred, minimum_reqs)
        filtered = [r for r in merged if not str(r).lower().startswith("databricks-mlops-stack")]
        # Preserve order while removing duplicates.
        filtered = list(dict.fromkeys(filtered))
        if not filtered:
            filtered = default_reqs
        logger.info("Final pip requirements count for model '%s': %d", model_name, len(filtered))

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example,
            code_paths=code_paths,
            pip_requirements=filtered,
        )

    def get_infer_mlflow_dependencies(self, code_paths: list[str], input_example, model, signature) -> tuple[list[str], list[str]]:
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
                gc.collect()
                inferred: list[str] = []
                req_file = Path(local_model_path) / "requirements.txt"
                if req_file.exists():
                    for line in req_file.read_text(encoding="utf-8").splitlines():
                        dep = line.strip()
                        if not dep or dep.startswith("#"):
                            continue
                        inferred.append(dep)
                if not inferred:
                    logger.warning(
                        "No requirements.txt inferred from temporary model path '%s'. "
                        "Calling mlflow.models.infer_pip_requirements fallback.",
                        local_model_path,
                    )
                    inferred = list(
                        mlflow.models.infer_pip_requirements(
                            model_uri=local_model_path,
                            flavor="sklearn",
                            fallback=default_reqs,
                        )
                    )
        except Exception as error:
            logger.exception(
                "Dependency inference failed. Falling back to default sklearn requirements. Error: %s",
                error,
            )
            inferred = default_reqs
        return default_reqs, inferred


build = ModelContractImpl() 
