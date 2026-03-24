import numpy as np   # type: ignore # noqa
import numpy.typing as npt   # type: ignore # noqa
from sklearn.base import BaseEstimator, TransformerMixin   # type: ignore # noqa

class ClipperTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to clip feature values based on a chosen strategy.

    Parameters
    ----------
    strategy : {"percentile", "median_range", "fixed_values"}, default="percentile"
        The clipping strategy to use.
    pct_lower : float in (0,100), default=1.0
        Lower percentile for clipping (used when strategy="percentile").
    pct_upper : float in (0,100), default=99.0
        Upper percentile for clipping (used when strategy="percentile").
    fixed_lower : Any float or None, default=None
        Lower fixed value for clipping (used when strategy="fixed_values").
    fixed_upper : Any float or None, default=None
        Upper fixed value for clipping (used when strategy="fixed_values").
    """
    def __init__(
            self,
            strategy: str ="percentile",
            pct_lower: float = 1.0,
            pct_upper: float = 99.0,
            fixed_lower: float | None = None,
            fixed_upper: float | None = None,
        ) -> None:

        self.strategy = strategy
        self.pct_lower = pct_lower
        self.pct_upper = pct_upper
        self.fixed_lower = fixed_lower
        self.fixed_upper = fixed_upper

        return

    def fit(
            self,
            X: npt.NDArray,
            y: npt.NDArray | None = None, # needed for scikit compability
        ) -> TransformerMixin:

        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.clip_values_ = np.zeros((self.n_features_in_, 2))

        if self.strategy == "percentile":
            for i in range(self.n_features_in_):
                col = X[:, i]
                col = col[np.isfinite(col)]
                lower = np.percentile(col, self.pct_lower)
                upper = np.percentile(col, self.pct_upper)
                if not np.isfinite(lower):
                    lower = np.nanmin(col)
                if not np.isfinite(upper):
                    upper = np.nanmax(col)
                self.clip_values_[i, :] = [lower, upper]

        elif self.strategy == "median_range":
            for i in range(self.n_features_in_):
                col = X[:, i]
                col = col[np.isfinite(col)]
                med = np.median(col)
                rng = (np.nanmax(col) - np.nanmin(col)) / 2.0
                self.clip_values_[i, :] = [med - rng, med + rng]

        elif self.strategy == "fixed_values":
            for i in range(self.n_features_in_):
                if self.fixed_upper is None:
                    self.fixed_upper = np.inf
                if self.fixed_lower is None:
                    self.fixed_lower = -np.inf
                self.clip_values_[i, :] = [self.fixed_lower, self.fixed_upper]

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return self

    def transform(
            self,
            X: npt.NDArray,
        ) -> npt.NDArray:

        X = np.asarray(X, dtype=float)
        X_clipped = np.empty_like(X)
        for i in range(self.n_features_in_):
            low, high = self.clip_values_[i]
            X_clipped[:, i] = np.clip(X[:, i], low, high)

        return X_clipped

    def get_feature_names_out(
            self,
            input_features: npt.NDArray[np.str_] | None = None,
        ) -> npt.NDArray[np.str_]:

        if input_features is not None:
            return np.array(input_features)

        return np.array([f"feature_{i}" for i in range(self.n_features_in_)])