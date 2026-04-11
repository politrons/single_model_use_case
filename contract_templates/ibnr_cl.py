"""
Chain Ladder regression model, sklearn-compatible.

Classes
-------
ChainLadderModel   - sklearn BaseEstimator implementing the Chain Ladder method.
"""

import logging
from typing import Any

import numpy as np  # type: ignore # noqa
import pandas as pd  # type: ignore # noqa

from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore # noqa

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Catalogs and constants
# -----------------------------------------------------------------------------

CONFIG_OCCURRENCE_DATE_COL = "occurrence_date_col"

_DEFAULT_OCCURRENCE_DATE_COL = "treatment_date"

_DEFAULT_CUMULATIVE_IDENTIFIER = "cumulative"

_DEFAULT_LAG_COL = "lag"

# ---------------------------------------------------------------------------
# 1. ChainLadderModel
# ---------------------------------------------------------------------------


class ChainLadderModel(BaseEstimator, RegressorMixin):
    """
    Chain Ladder development-factor model, compatible with scikit-learn.

    Parameters
    ----------
    config : dict[str, Any]
        Model-level configuration. Recognised keys:
        - ``occurrence_date_col`` (str): column name for occurrence date
          (default ``"treatment_date"``).
        - ``cumulative_identifier`` (str): substring to identify the
          cumulative column among input features (default ``"cumulative"``).
        - ``lag_col`` (str): name of the lag / development period column
          (default ``"lag"``).
    random_state : int
        Unused — kept for interface compatibility.
    params : dict
        Reserved for future / framework-level parameters.
    extra_params : dict
        Reserved for future / framework-level parameters.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        random_state: int = 42,
        params: dict | None = None,
        extra_params: dict | None = None,
    ) -> None:

        self.config: dict[str, Any] = config or {}
        logger.info(f"config: {self.config}")
        self.params: dict = params or {}
        logger.info(f"params: {self.params}")
        self.extra_params: dict = extra_params or {}
        logger.info(f"extra params: {self.extra_params}")

        self.random_state: int = random_state

        # --- resolve config fields -----------------------------------------
        self.occurrence_date_col: str = self.config.get(CONFIG_OCCURRENCE_DATE_COL, _DEFAULT_OCCURRENCE_DATE_COL)
        logger.info(f"occurrence date column: '{self.occurrence_date_col}'")

        self.cumulative_identifier: str = _DEFAULT_CUMULATIVE_IDENTIFIER
        logger.info(f"cumulative identifier: '{self.cumulative_identifier}'")

        self.lag_col: str = _DEFAULT_LAG_COL
        logger.info(f"lag column: '{self.lag_col}'")

        # --- learned state (populated by fit) ------------------------------
        self.cumulative_col_: str | None = None
        self.segmentation_cols_: list[str] = []
        self.dev_factors_: pd.DataFrame | None = None
        self.is_fitted_: bool = False

        return

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        **extra_params: Any,
    ) -> None:
        """
        Compute development factors from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain:
            - Exactly one column whose name includes the cumulative
              identifier (default ``"cumulative"``)
            - The lag column (default ``"lag"``)
            - The occurrence-date column specified in config
            - Optionally, additional segmentation columns
        y : pd.Series | None
            Ignored — the target is implicit in the cumulative triangle.
        **extra_params
            Currently unused; reserved for framework compatibility.

        Returns
        -------
        self
        """
        df_train = X.copy()

        # --- identify cumulative column -----------------------------------
        cumulative_cols = [c for c in df_train.columns if self.cumulative_identifier in c]
        if len(cumulative_cols) != 1:
            raise ValueError(f"Expected exactly one column containing '{self.cumulative_identifier}' in its name, found {len(cumulative_cols)}: {cumulative_cols}")
        self.cumulative_col_ = cumulative_cols[0]
        logger.info(f"cumulative column: '{self.cumulative_col_}'")

        # --- identify segmentation columns --------------------------------
        reserved_cols = {self.cumulative_col_, self.lag_col, self.occurrence_date_col}
        self.segmentation_cols_ = [c for c in df_train.columns if c not in reserved_cols]
        logger.info(f"segmentation columns: {self.segmentation_cols_}")

        # --- sort -----------------------------------------------------------
        sort_keys = self.segmentation_cols_ + [self.occurrence_date_col, self.lag_col]
        df_train = df_train.sort_values(by=sort_keys)

        # --- next-period cumulative ----------------------------------------
        group_keys = self.segmentation_cols_ + [self.occurrence_date_col]
        df_train["Next_Cumulative"] = df_train.groupby(group_keys)[self.cumulative_col_].shift(-1)

        # --- development factors (only where denominator > 0) --------------
        mask = df_train[self.cumulative_col_] > 0
        df_train["Development Factor"] = np.nan
        df_train.loc[mask, "Development Factor"] = df_train.loc[mask, "Next_Cumulative"] / df_train.loc[mask, self.cumulative_col_]

        # --- average development factors per lag (+ segmentation) ----------
        df_train = df_train.sort_values(
            by=self.segmentation_cols_ + [self.occurrence_date_col, self.lag_col],
            ascending=False,
        )

        factor_group_keys = [self.lag_col] + self.segmentation_cols_
        df_factors = df_train.groupby(factor_group_keys)["Development Factor"].mean().reset_index()
        df_factors.rename(
            columns={"Development Factor": "Avg Development Factor"},
            inplace=True,
        )
        df_factors["Avg Development Factor"] = df_factors["Avg Development Factor"].fillna(1.0)

        # --- cumulative factors --------------------------------------------
        df_factors = df_factors.sort_values(
            by=self.segmentation_cols_ + [self.lag_col],
            ascending=False,
        )

        if not self.segmentation_cols_:
            df_factors["cumulative Factor"] = df_factors["Avg Development Factor"].cumprod()
        else:
            df_factors["cumulative Factor"] = df_factors.groupby(self.segmentation_cols_)["Avg Development Factor"].cumprod()

        df_factors = df_factors.sort_values(
            by=self.segmentation_cols_ + [self.lag_col],
        )

        self.dev_factors_ = df_factors
        self.is_fitted_ = True

        logger.info(f"Chain ladder fitted — {len(df_factors)} factor rows, {self.lag_col} range [{df_factors[self.lag_col].min()}, {df_factors[self.lag_col].max()}]")

        return

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Apply cumulative development factors to produce ultimate predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain the lag column, the cumulative column, and any
            segmentation columns used during ``fit()``.

        Returns
        -------
        np.ndarray
            1-D array of predicted ultimate values.
        """
        if not self.is_fitted_ or self.dev_factors_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        merge_keys = [self.lag_col] + self.segmentation_cols_

        df = X.copy()
        df = df.merge(
            self.dev_factors_,
            on=merge_keys,
            how="left",
        ).fillna(1.0)

        predictions = (df[self.cumulative_col_] * df["cumulative Factor"]).to_numpy(dtype=np.float64)

        return predictions

    def get_params(
        self,
        deep: bool = True,
    ) -> dict[str, Any]:
        logger.info("getting params ... ")
        return {
            "config": self.config,
            "random_state": self.random_state,
            "params": self.params,
            "extra_params": self.extra_params,
        }


# ---------------------------------------------------------------------------
# 2. Builder function
# ---------------------------------------------------------------------------


def _build_cluster_model(
    config: dict,
    random_state: int = 42,
    base_params: dict | None = None,
    extra_params: dict | None = None,
) -> ChainLadderModel:
    """
    Factory that builds a ChainLadderModel.

    Unlike the TF variant there is no need for a
    ``TransformedTargetRegressor`` or ``RobustScaler`` pipeline because
    the Chain Ladder method operates on raw cumulative values and
    multiplicative factors.

    Parameters
    ----------
    config, random_state, base_params, extra_params
        Forwarded to ``ChainLadderModel``.

    Returns
    -------
    ChainLadderModel (unfitted)
    """
    return ChainLadderModel(
        config=config,
        random_state=random_state,
        params=base_params or {},
        extra_params=extra_params or {},
    )
