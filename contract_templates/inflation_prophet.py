import logging
from typing import Any
import numpy as np  # type: ignore # noqa
import pandas as pd  # type: ignore # noqa
from dateutil.relativedelta import relativedelta  # type: ignore # noqa
from prophet import Prophet  # type: ignore # noqa
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore # noqa
from sklearn.pipeline import Pipeline  # type: ignore # noqa

class ProphetModel(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        config: dict[str, Any],
        random_state: int,
        params: dict,
        extra_params: dict,
    ) -> None:
        
        ###### Initialize the model here, through the config
        ### most likley the config will be the 'clusters' property on the models_created from run_config
        ### that contains, for each cluster, the actual config for the cluster (you can find by the cluster_id)

        ### the segmentation column will be only one, the cluster id (but still a list) ... but this will be passed at the MultiClusterWrapper level

        ### fit and predict will need to be refactored according to the signatures below (train needs to be renamed to fit)
        ### since the properties will be parsed during init, you can access them through 'self'
        ### regressors data will be already on the X, so you can access it (check how I did for neuralprophet)

        return
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **extra_params,
    ) -> None:
        
        return
    
    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        
        return np.zeros(1)

    # def train(self, df_past: pd.DataFrame, temporal_column: str, target_column: str) -> None:
    #     min_date = df_past[temporal_column].min()
    #     max_date = df_past[temporal_column].max()
    #     total_nb_train_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1

    #     cutoff_date = max_date - relativedelta(months=self.config.nb_months_censored)
    #     df_past.loc[df_past[temporal_column] > cutoff_date, target_column] = None

    #     self.model = Prophet(
    #         yearly_seasonality=False,
    #         weekly_seasonality=False,
    #         daily_seasonality=False,
    #         changepoint_range=1.0,
    #         n_changepoints=int(self.config.changepoint_nb_per_year * (total_nb_train_months - self.config.nb_months_censored) / 12),
    #         changepoint_prior_scale=self.config.changepoint_prior_scale,
    #     )

    #     if self.config.seasonality_period:
    #         self.model.add_seasonality(
    #             name="seasonality",
    #             period=self.config.seasonality_period,
    #             fourier_order=self.config.seasonality_fourier_order,
    #         )

    #     if self.config.country_holidays:
    #         self.model.add_country_holidays(country_name=self.config.country_holidays)

    #     if self.config.shocks:
    #         shocks = pd.DataFrame(columns=["holiday", "ds", "lower_window", "ds_upper"])
    #         for name, shock_window in self.config.shocks.items():
    #             shocks.loc[len(shocks)] = {
    #                 "holiday": f"shock_{name}",
    #                 "ds": pd.to_datetime(shock_window[0]),
    #                 "lower_window": 0,
    #                 "ds_upper": pd.to_datetime(shock_window[1]),
    #             }
    #         shocks["upper_window"] = (shocks["ds_upper"] - shocks["ds"]).dt.days  # TODO : unsure here

    #         self.model.holidays = shocks
    #         self.model.holidays_prior_scale = self.config.shock_prior_scale

    #     if self.config.regressors:
    #         for regressor, regressor_prior_scale in self.config.regressors.items():
    #             self.model.add_regressor(regressor, prior_scale=regressor_prior_scale)

    #     prophet_df = df_past.reset_index().rename(columns={temporal_column: "ds", target_column: "y"})
    #     self.model.fit(prophet_df)

    # def predict(
    #     self,
    #     df_futur: pd.DataFrame,
    #     temporal_column: str,
    #     target_column: str,
    #     n_months_to_forecast: int,
    #     verbose: bool = False,
    # ) -> pd.DataFrame:
    #     future = self.model.make_future_dataframe(periods=n_months_to_forecast, freq="M")
    #     future["ds"] = future["ds"].apply(lambda x: (x.replace(day=1) if x.day < 15 else (x + pd.offsets.MonthBegin(1)).replace(day=1)))
    #     if self.config.regressors:
    #         future = pd.merge(
    #             future,
    #             df_futur.rename(columns={temporal_column: "ds"}),
    #             on="ds",
    #             how="left",
    #         )

    #     forecast = self.model.predict(future)

    #     if verbose:
    #         self.model.plot_components(forecast)

    #     forecast.rename(
    #         columns={
    #             "ds": temporal_column,
    #             "yhat": target_column + "_pred",
    #             "yhat_lower": target_column + "_pred_lower",
    #             "yhat_upper": target_column + "_pred_upper",
    #         },
    #         inplace=True,
    #     )

    #     return forecast

def _build_cluster_model(
    config: dict,
    random_state: int,
    base_params: dict,
    extra_params: dict,
) -> Pipeline:

    ph_model = ProphetModel(
        config=config,
        random_state=random_state,
        params=base_params,
        extra_params=extra_params,
    )

    pipeline = Pipeline(steps=[
        ("passthrough", "passthrough"),
        ("model", ph_model)
    ]) 

    return pipeline
