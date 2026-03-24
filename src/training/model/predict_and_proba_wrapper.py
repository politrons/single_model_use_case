from __future__ import annotations

import json
from typing import Any

import numpy as np  # type: ignore # noqa
import pandas as pd  # type: ignore # noqa
from sklearn.base import BaseEstimator  # type: ignore # noqa


class PredictAndProbaWrapper(BaseEstimator):
    """Wrapper that exposes predict() with prediction + probability output."""

    def __init__(self, model_name: str, internal_model: Any) -> None:
        self.model_name = model_name
        self.internal_model_key = f"internal_{model_name}"
        self._internal_model = internal_model
        # Keep an explicit map for traceability/debuggability.
        self.internal_models = {self.internal_model_key: internal_model}

    def fit(self, X, y, *args, **kwargs):
        self._internal_model.fit(X, y, *args, **kwargs)
        return self

    def predict(self, X, *args, **kwargs):
        preds = self._internal_model.predict(X, *args, **kwargs)
        probs = self._internal_model.predict_proba(X)
        n_rows = len(X)
        pred_series = to_prediction_series(preds, n_rows)
        proba_series = to_probability_series(probs, n_rows)
        return pd.DataFrame({
            "prediction": pred_series,
            "prediction_proba": proba_series,
        })

    @property
    def internal_model(self) -> Any:
        return self._internal_model

    def __getattr__(self, item: str):
        return getattr(self._internal_model, item)


def to_prediction_series(values: Any, n_rows: int) -> pd.Series:
    # Normalize heterogeneous predict outputs into a single 1D pandas Series.
    # We accept Series/DataFrame/list/ndarray/scalar and enforce "one prediction per input row".
    if isinstance(values, pd.Series):
        # Already 1D; just drop any previous index to align with current batch rows.
        out = values.reset_index(drop=True)
    elif isinstance(values, pd.DataFrame):
        # Prefer explicit "prediction" column; otherwise allow a single-column DataFrame.
        if "prediction" in values.columns:
            out = values["prediction"].reset_index(drop=True)
        elif values.shape[1] == 1:
            out = values.iloc[:, 0].reset_index(drop=True)
        else:
            raise ValueError("Predictions DataFrame must contain 'prediction' or exactly one column.")
    else:
        # Convert array-like inputs to numpy to handle scalar/1D/2D shapes uniformly.
        arr = np.asarray(values)
        if arr.ndim == 0:
            # Scalar prediction: broadcast to all rows.
            out = pd.Series([arr.item()] * n_rows)
        elif arr.ndim == 1:
            # Standard case: one prediction per row.
            out = pd.Series(arr.tolist())
        elif arr.ndim == 2 and arr.shape[1] == 1:
            # Column vector -> flatten to 1D.
            out = pd.Series(arr.reshape(-1).tolist())
        else:
            # Multi-column predictions are ambiguous here; probabilities must go through predict_proba flow.
            raise ValueError("Predictions must be 1D. Use prediction_method='predict_proba' for probabilities.")

    # Hard fail on cardinality mismatch to avoid silent row misalignment in downstream tables.
    if len(out) != n_rows:
        raise ValueError(f"Prediction length mismatch. Expected {n_rows}, got {len(out)}.")
    return out

def to_probability_series(values: Any, n_rows: int) -> pd.Series:
    # Normalize probability outputs into one storable value per row.
    # Multi-class vectors are serialized as JSON strings to preserve full information.
    if isinstance(values, pd.Series):
        # Already row-wise; serialize each cell to a stable string representation.
        out = values.reset_index(drop=True).map(_serialize_probability_value)
    elif isinstance(values, pd.DataFrame):
        # Prefer explicit "prediction_proba"; otherwise allow single-column or row-wise serialization.
        if "prediction_proba" in values.columns:
            out = values["prediction_proba"].reset_index(drop=True).map(_serialize_probability_value)
        elif values.shape[1] == 1:
            out = values.iloc[:, 0].reset_index(drop=True).map(_serialize_probability_value)
        else:
            # Multiple columns usually mean class probabilities split by column; keep the full row as JSON.
            out = values.apply(lambda row: _serialize_probability_value(row.tolist()), axis=1)
    else:
        # Handle scalar/list/ndarray uniformly by shape.
        arr = np.asarray(values, dtype=object)
        if arr.ndim == 0:
            # Scalar probability: broadcast to all rows.
            out = pd.Series([_serialize_probability_value(arr.item())] * n_rows)
        elif arr.ndim == 1:
            # One probability value/object per row.
            out = pd.Series([_serialize_probability_value(v) for v in arr.tolist()])
        elif arr.ndim == 2:
            # Typical predict_proba output (n_rows x n_classes): serialize each row vector.
            out = pd.Series([_serialize_probability_value(row.tolist()) for row in arr])
        else:
            raise ValueError("Probability output must be 1D or 2D.")

    # Hard fail on cardinality mismatch to prevent writing corrupted baseline/monitoring rows.
    if len(out) != n_rows:
        raise ValueError(f"Probability length mismatch. Expected {n_rows}, got {len(out)}.")
    return out


def _serialize_probability_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, pd.Series):
        value = value.tolist()
    try:
        return json.dumps(value)
    except Exception:
        return str(value)
