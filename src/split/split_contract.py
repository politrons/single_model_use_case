from abc import ABC, abstractmethod
from typing import Any, TypedDict, Optional

import pandas as pd   # type: ignore # noqa

class SplitResult(TypedDict):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_val: Optional[pd.DataFrame]
    y_val: Optional[pd.Series]

class SplitContract(ABC):

    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series, split_kwargs: dict[str, Any]) -> SplitResult:
        pass
