from abc import ABC, abstractmethod
from typing import Any

import pandas as pd   # type: ignore # noqa


class Split(ABC):

    @abstractmethod
    def split(self, x: pd.DataFrame, y: pd.Series, split_kwargs: dict[str, Any]) -> dict[str, Any]:
        pass
