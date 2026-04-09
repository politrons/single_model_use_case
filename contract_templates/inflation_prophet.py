import logging
from datetime import datetime
from typing import Any

import numpy as np  # type: ignore # noqa
import pandas as pd  # type: ignore # noqa
from dateutil.relativedelta import relativedelta  # type: ignore # noqa
from prophet import Prophet  # type: ignore # noqa
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore # noqa

LOG = logging.getLogger(__name__)


class ProphetModel(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        config: dict[str, Any],
        random_state: int,
        params: dict,
        extra_params: dict,
    ) -> None:
        self.config = config or {}
        self.random_state = random_state
        self.params = params or {}
        self.extra_params = extra_params or {}

        self.feature_columns: list[str] = list(self.extra_params.get("feature_columns", []))
        self.temporal_column: str = str(self.extra_params.get("temporal_reference_column", "ds"))

        self._cluster_key: tuple | None = None
        self._active_model_config: dict[str, Any] = {}
        self._regressors: list[str] = []
        self.model: Prophet | None = None

    def set_cluster_key(self, key: tuple) -> None:
        self._cluster_key = key

    def _cluster_key_aliases(self) -> list[str]:
        if self._cluster_key is None:
            return []
        raw = self._cluster_key if isinstance(self._cluster_key, tuple) else (self._cluster_key,)
        aliases = {
            str(self._cluster_key),
            str(raw),
            "__".join(str(x) for x in raw),
        }
        if len(raw) == 1:
            aliases.add(str(raw[0]))
        return list(aliases)

    def _resolve_model_config(self) -> dict[str, Any]:
        base = self.config if isinstance(self.config, dict) else {}
        resolved: dict[str, Any] = {}

        default_cfg = base.get("default_model_config")
        if isinstance(default_cfg, dict):
            resolved.update(default_cfg)

        cluster_map = base.get("cluster_model_config_map")
        if isinstance(cluster_map, dict):
            for alias in self._cluster_key_aliases():
                maybe_cluster_cfg = cluster_map.get(alias)
                if isinstance(maybe_cluster_cfg, dict):
                    resolved.update(maybe_cluster_cfg)
                    break

        for k, v in base.items():
            if k in {"default_model_config", "cluster_model_config_map"}:
                continue
            resolved.setdefault(k, v)

        return resolved

    def _to_feature_dataframe(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            df = x.copy()
        else:
            arr = np.asarray(x)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if self.feature_columns and len(self.feature_columns) == arr.shape[1]:
                cols = self.feature_columns
            else:
                cols = [f"feature_{i}" for i in range(arr.shape[1])]
            df = pd.DataFrame(arr, columns=cols)

        if self.temporal_column not in df.columns:
            raise ValueError(
                f"Temporal column '{self.temporal_column}' not found in features. "
                f"Columns: {list(df.columns)}"
            )

        df[self.temporal_column] = pd.to_datetime(df[self.temporal_column], errors="coerce")
        if df[self.temporal_column].isna().any():
            raise ValueError(f"Temporal column '{self.temporal_column}' contains invalid dates")

        return df.reset_index(drop=True)

    def _build_prophet(self, model_cfg: dict[str, Any], df: pd.DataFrame) -> Prophet:
        changepoint_nb_months_censored = int(model_cfg.get("changepoint_nb_months_censored", 0) or 0)
        changepoint_nb_per_year = model_cfg.get("changepoint_nb_per_year")
        n_changepoints = model_cfg.get("n_changepoints")
        if n_changepoints is None and changepoint_nb_per_year:
            max_date = df[self.temporal_column].max()
            min_date = df[self.temporal_column].min()
            if isinstance(max_date, datetime) and isinstance(min_date, datetime):
                effective_max = max_date - relativedelta(months=changepoint_nb_months_censored)
                total_nb_months = (effective_max.year - min_date.year) * 12 + (effective_max.month - min_date.month) + 1
                total_nb_months = max(1, total_nb_months)
                n_changepoints = int(float(changepoint_nb_per_year) * total_nb_months / 12)

        prophet_kwargs = {
            "yearly_seasonality": bool(model_cfg.get("yearly_seasonality", False)),
            "weekly_seasonality": bool(model_cfg.get("weekly_seasonality", False)),
            "daily_seasonality": bool(model_cfg.get("daily_seasonality", False)),
            "changepoint_range": float(model_cfg.get("changepoint_range", 1.0)),
            "changepoint_prior_scale": float(model_cfg.get("changepoint_prior_scale", 0.05)),
        }
        if n_changepoints is not None:
            prophet_kwargs["n_changepoints"] = int(n_changepoints)

        model = Prophet(**prophet_kwargs)

        seasonality_period = model_cfg.get("seasonality_period")
        seasonality_fourier_order = model_cfg.get("seasonality_fourier_order")
        if seasonality_period and seasonality_fourier_order:
            model.add_seasonality(
                name="custom_seasonality",
                period=float(seasonality_period),
                fourier_order=int(seasonality_fourier_order),
            )

        country_holidays = model_cfg.get("country_holidays")
        if country_holidays:
            model.add_country_holidays(country_name=str(country_holidays))

        regressors = model_cfg.get("regressors")
        self._regressors = []
        if isinstance(regressors, dict):
            for regressor_name, prior_scale in regressors.items():
                kwargs: dict[str, Any] = {}
                try:
                    kwargs["prior_scale"] = float(prior_scale)
                except Exception:
                    pass
                model.add_regressor(str(regressor_name), **kwargs)
                self._regressors.append(str(regressor_name))

        return model

    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **extra_params,
    ) -> "ProphetModel":
        _ = extra_params
        df = self._to_feature_dataframe(x)
        target = pd.Series(y).reset_index(drop=True)

        if len(df) != len(target):
            raise ValueError(f"X/y length mismatch ({len(df)} != {len(target)})")

        model_cfg = self._resolve_model_config()
        self._active_model_config = model_cfg
        self.model = self._build_prophet(model_cfg, df)

        prophet_df = pd.DataFrame(
            {
                "ds": df[self.temporal_column],
                "y": pd.to_numeric(target, errors="coerce"),
            }
        )
        for regressor in self._regressors:
            if regressor not in df.columns:
                raise ValueError(
                    f"Regressor '{regressor}' declared in config but missing from features. "
                    f"Columns: {list(df.columns)}"
                )
            prophet_df[regressor] = pd.to_numeric(df[regressor], errors="coerce")

        months_censored = int(model_cfg.get("changepoint_nb_months_censored", 0) or 0)
        if months_censored > 0:
            cutoff_date = prophet_df["ds"].max() - relativedelta(months=months_censored)
            prophet_df.loc[prophet_df["ds"] > cutoff_date, "y"] = None

        self.model.fit(prophet_df)
        return self

    def predict(
        self,
        x: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ProphetModel must be fitted before predict().")

        df = self._to_feature_dataframe(x)
        future = pd.DataFrame({"ds": df[self.temporal_column]})
        for regressor in self._regressors:
            if regressor not in df.columns:
                raise ValueError(
                    f"Regressor '{regressor}' declared in config but missing from prediction features."
                )
            future[regressor] = pd.to_numeric(df[regressor], errors="coerce")

        forecast = self.model.predict(future)
        if "yhat" not in forecast.columns:
            raise ValueError("Prophet forecast does not contain 'yhat'")
        return pd.to_numeric(forecast["yhat"], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _build_cluster_model(
    config: dict,
    random_state: int,
    base_params: dict,
    extra_params: dict,
) -> ProphetModel:
    _ = base_params
    return ProphetModel(
        config=config,
        random_state=random_state,
        params=base_params,
        extra_params=extra_params,
    )
