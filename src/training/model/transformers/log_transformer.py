import numpy as np   # type: ignore # noqa
import numpy.typing as npt   # type: ignore # noqa
from sklearn.base import BaseEstimator, TransformerMixin   # type: ignore # noqa
from sklearn.preprocessing import FunctionTransformer   # type: ignore # noqa

def log_func(
        x: npt.NDArray,
    ) -> npt.NDArray:
    return np.log1p(x)

def log_func_inv(
        x: npt.NDArray,
    ) -> npt.NDArray:
    return np.expm1(x)

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:

        self.transformer = FunctionTransformer(
            func=log_func,
            inverse_func=log_func_inv,
            feature_names_out="one-to-one",
            validate=False
        )

        return

    def fit(
            self,
            X: npt.NDArray,
            y: npt.NDArray | None = None, # needed for scikit compability
        ) -> TransformerMixin:

        self.feature_names_in_ = getattr(X, "columns", None)
        self.transformer.fit(X)

        return self

    def transform(
            self,
            X: npt.NDArray,
        ) -> npt.NDArray:

        return self.transformer.transform(X)
    
    def inverse_transform(
            self,
            X: npt.NDArray,
        ) -> npt.NDArray:

        return self.transformer.inverse_transform(X)

    def get_feature_names_out(
            self,
            input_features: npt.NDArray[np.str_] | None = None,
        ) -> npt.NDArray[np.str_]:

        if input_features is not None:
            return np.array(input_features)

        return np.array([f"feature_{i}" for i in range(self.n_features_in_)])
