"""
multi_cluster_wrapper.py

MultiClusterWrapper: trains and serves one internal model per unique
combination of segment columns, producing a single sklearn-compatible
artefact. The internal model type is controlled by a factory callable,
making the wrapper reusable across TensorFlow, Prophet, or any other
backend.
"""

import gc
import logging
from collections.abc import Callable  # type: ignore # noqa
from typing import Any

import numpy as np  # type: ignore # noqa
import pandas as pd  # type: ignore # noqa
import psutil  # type: ignore # noqa
from joblib import Parallel, delayed  # type: ignore # noqa
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore # noqa

logger = logging.getLogger(__name__)


class MultiClusterWrapper(BaseEstimator, RegressorMixin):
    """
    Trains and serves one internal model per unique combination of
    `segment_columns`, producing a single sklearn-compatible artefact.

    Parameters
    ----------
    numerical_features : list[str]
        Feature columns used for training/prediction.
    segment_columns : list[str]
        Columns whose cross-product defines the segments.
    model_factory : Callable
        A callable with signature:
            model_factory(config, random_state, base_params, extra_params)
                -> fitted-able sklearn-compatible estimator
        Called once per segment during fit() to construct the internal model.
        Example: _build_tf_segment_model from tf_model.py.
    is_tensorflow : bool
        Set to True when model_factory produces TensorFlow-backed models.
        Enables TF-specific optimisations during predict:
          - tf.config.run_functions_eagerly(False)
          - input conversion to tf.constant before predict
          - keras.backend.clear_session() after all segments are fitted
    config : dict
        Either:
        1) a single model config forwarded to every segment, or
        2) a dict where keys are segment ids and values are per-segment configs.
    random_state : int
        Forwarded to model_factory.
    base_params : dict
        Forwarded to model_factory.
    extra_params : dict
        Forwarded to model_factory.
    n_jobs : int
        Number of parallel workers used during fit (-1 = all cores).
    """

    def __init__(
        self,
        numerical_features: list[str],
        segment_columns: list[str],
        model_factory: Callable,
        is_tensorflow: bool,
        config: dict[str, Any],
        random_state: int,
        base_params: dict,
        extra_params: dict,
        n_jobs: int,
    ):
        """
        Store wrapper settings and initialize runtime state.

        Parameters are intentionally kept as plain serializable attributes so
        scikit-learn `get_params`/`set_params` can introspect the estimator.
        Fitted artifacts are stored later in `self.models_` during `fit()`.
        """
        self.numerical_features = numerical_features or []
        self.segment_columns = segment_columns or []
        self.model_factory = model_factory
        self.is_tensorflow = is_tensorflow
        self.config = config or {}
        self.random_state = random_state
        self.base_params = base_params or {}
        self.extra_params = extra_params or {}
        self.n_jobs = n_jobs

        # Populated by fit()
        self.models_: dict[tuple, Any] = {}
        self.target_column_: str | None = None
        self.number_clusters: int = -1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cluster_key(key) -> tuple:
        """Normalise a groupby key to a hashable tuple."""
        return key if isinstance(key, tuple) else (key,)

    @staticmethod
    def _cluster_key_aliases(key: tuple) -> list[str]:
        """
        Build equivalent string aliases for a segment key.

        This allows matching per-segment configuration maps that may use
        different key encodings (tuple string, joined key, or scalar key).
        """
        aliases = {str(key)}
        aliases.add("__".join(str(x) for x in key))
        if len(key) == 1:
            aliases.add(str(key[0]))
        return [x for x in aliases if x]

    @staticmethod
    def _looks_like_segment_config_map(cfg: dict[str, Any]) -> bool:
        """Return True when `cfg` appears to be a `segment -> config` dictionary."""
        if not cfg:
            return False
        return all(isinstance(v, dict) for v in cfg.values())

    def _config_for_param_logging(self) -> Any:
        """
        Return a compact `config` representation safe for MLflow param logging.

        MLflow model params have a max size constraint per value; large maps
        such as `cluster_model_config_map` can exceed that limit. This method
        keeps enough metadata for observability while preventing oversized
        payloads.
        """
        cfg = self.config
        if not isinstance(cfg, dict):
            return cfg

        cluster_map = cfg.get("cluster_model_config_map")
        if isinstance(cluster_map, dict):
            preview = [str(k) for k in list(cluster_map.keys())[:5]]
            return {
                "mode": "cluster_model_config_map",
                "cluster_count": len(cluster_map),
                "cluster_keys_preview": preview,
                "default_model_config": cfg.get("default_model_config", {}),
            }

        if self._looks_like_segment_config_map(cfg):
            preview = [str(k) for k in list(cfg.keys())[:5]]
            return {
                "mode": "segment_config_map",
                "cluster_count": len(cfg),
                "cluster_keys_preview": preview,
            }

        return cfg

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Expose estimator params with a compact `config` field for logging.

        We keep the runtime behavior unchanged (the full config is still held
        in `self.config`), but replace only the exported param view so tools
        like MLflow do not attempt to log very large dictionaries verbatim.
        """
        params = super().get_params(deep=deep)
        params["config"] = self._config_for_param_logging()
        return params

    def _resolve_config_for_segment(self, key: tuple) -> dict[str, Any]:
        """
        Resolve the effective model config for one segment key.

        Behavior:
        - If `self.config` is a single global config, return it as-is.
        - If `self.config` is a segment map, select the matching segment config.
        - If no match exists, return an empty config and log a warning.
        """
        cfg = self.config if isinstance(self.config, dict) else {}
        if not self._looks_like_segment_config_map(cfg):
            return cfg

        for alias in self._cluster_key_aliases(key):
            maybe_cfg = cfg.get(alias)
            if isinstance(maybe_cfg, dict):
                return maybe_cfg

        logger.warning("No per-segment config found for key %s; using empty config.", key)
        return {}

    def _fit_one_cluster(
        self,
        key: tuple,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[tuple, Any]:
        """Build and fit a single cluster model; designed for parallel use."""
        resolved_config = self._resolve_config_for_segment(key)
        model = self.model_factory(
            resolved_config,
            self.random_state,
            self.base_params,
            self.extra_params,
        )
        # Optional hook for models that need cluster-aware runtime config.
        if hasattr(model, "set_cluster_key"):
            model.set_cluster_key(key)
        elif hasattr(model, "named_steps"):
            maybe_step_model = model.named_steps.get("model")
            if hasattr(maybe_step_model, "set_cluster_key"):
                maybe_step_model.set_cluster_key(key)
        model.fit(X, y)
        logger.info(f"Finished fitting cluster: {key}")
        return key, model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, y: pd.Series | str) -> "MultiClusterWrapper":
        """
        Fit one internal model per unique combination of `segment_columns`.

        Parameters
        ----------
        df : pd.DataFrame
            Full training dataframe. Must contain `segment_columns`,
            `numerical_features`, and the target column.
        y : pd.Series or str
            Target values, or the name of the target column in `df`.

        Returns
        -------
        self
        """
        if self.model_factory is None:
            raise ValueError("model_factory must be provided before calling fit().")

        if isinstance(y, str):
            target = df[y]
            self.target_column_ = y
        else:
            target = y
            self.target_column_ = target.name

        missing_seg = [c for c in self.segment_columns if c not in df.columns]
        if missing_seg:
            raise ValueError(f"segment columns not found in df: {missing_seg}")

        missing_feat = [c for c in self.numerical_features if c not in df.columns]
        if missing_feat:
            raise ValueError(f"Numerical features not found in df: {missing_feat}")

        groups = list(df.groupby(self.segment_columns, sort=False))
        logger.info("Fitting %d cluster models with n_jobs=%d …", len(groups), self.n_jobs)

        results: list[tuple[tuple, Any]] = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._fit_one_cluster)(
                self._cluster_key(key),
                group[self.numerical_features],
                target.loc[group.index],
            )
            for key, group in groups
        )

        self.models_ = dict(results)
        self.number_clusters = len(self.models_)
        logger.info("Fitting complete. Clusters stored: %d", len(self.models_))

        # TF-specific: clear Keras session state accumulated during training.
        # Safe to call here because all models are already fitted and stored;
        # none rely on a persistent Keras session after fit() completes.
        if self.is_tensorflow:
            try:
                from tensorflow import keras  # type: ignore[import-untyped]

                keras.backend.clear_session()
                logger.info("Keras backend session cleared after fitting.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not clear Keras session: %s", exc)

        gc.collect()

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for all rows in `df`.

        Rows are grouped by `segment_columns`; each group is dispatched to
        the corresponding fitted cluster model. The original row order is
        preserved in the returned array.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain `segment_columns` and `numerical_features`.
            May contain any subset of the clusters seen during fit.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        KeyError
            If a cluster in `df` was not seen during fit.
        ValueError
            If required columns are missing from `df`.
        """
        if not self.models_:
            raise RuntimeError("MultiClusterWrapper is not fitted yet. Call fit() first.")

        missing_seg = [c for c in self.segment_columns if c not in df.columns]
        if missing_seg:
            raise ValueError(f"segment columns not found in df: {missing_seg}")

        # TF-specific: ensure graph mode is active to reduce retracing.
        if self.is_tensorflow:
            try:
                import tensorflow as tf  # type: ignore[import-untyped]

                tf.config.run_functions_eagerly(False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not set TF eager mode: %s", exc)

        process = psutil.Process()
        predictions = pd.Series(index=df.index, dtype=float)

        if self.is_tensorflow:
            import tensorflow as tf  # type: ignore[import-untyped]

        for key, group in df.groupby(self.segment_columns, sort=False):
            seg_key = self._cluster_key(key)

            if seg_key not in self.models_:
                raise KeyError(f"Cluster {seg_key} was not seen during training. Cannot generate predictions for unseen clusters.")

            X = group[self.numerical_features]

            preds = self.models_[seg_key].predict(X)
            predictions.loc[group.index] = preds

            mem_gb = process.memory_info().rss / 1024**3
            logger.info("segment %s — Memory: %.2f GB", seg_key, mem_gb)

        gc.collect()

        # Restore original row order
        return predictions.loc[df.index].values

    def compact_for_serialization(self) -> None:
        """
        Run best-effort model compaction hooks before serialization.

        The wrapper traverses direct estimators and common nested containers
        (`named_steps`, `regressor`, `estimator`) and calls
        `prepare_for_serialization()` when available. Failures are swallowed on
        purpose because compaction is an optimization and must not block model
        logging.
        """
        for maybe_estimator in self.models_.values():
            if hasattr(maybe_estimator, "prepare_for_serialization"):
                try:
                    maybe_estimator.prepare_for_serialization()
                except Exception:
                    pass
            named_steps = getattr(maybe_estimator, "named_steps", None)
            if isinstance(named_steps, dict):
                step_model = named_steps.get("model")
                if step_model is not None and hasattr(step_model, "prepare_for_serialization"):
                    try:
                        step_model.prepare_for_serialization()
                    except Exception:
                        pass
            for nested_attr in ("regressor", "regressor_", "estimator", "estimator_"):
                nested = getattr(maybe_estimator, nested_attr, None)
                if nested is None:
                    continue
                nested_steps = getattr(nested, "named_steps", None)
                if isinstance(nested_steps, dict):
                    nested_model = nested_steps.get("model")
                    if nested_model is not None and hasattr(nested_model, "prepare_for_serialization"):
                        try:
                            nested_model.prepare_for_serialization()
                        except Exception:
                            pass
        gc.collect()
