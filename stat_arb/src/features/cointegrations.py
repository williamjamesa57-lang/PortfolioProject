from statsmodels.regression.linear_model import RegressionResults
from statsmodels.tsa.stattools import coint, adfuller
from utils import data_loader
from typing import Any

import statsmodels.api as sm
import pandas as pd
import numpy as np


def determine_top_cointegrated_pairs(given_stack: pd.DataFrame, n: int) -> pd.DataFrame:
    given_stack.sort_values(by=["p"], inplace=True)
    corr_stack_s = given_stack.iloc[0:n]
    return corr_stack_s


class CointegrationEngine:
    def __init__(
            self,
            data_loader_source: data_loader.DataLoader
    ) -> None:
        self.__data_loader: data_loader.DataLoader = data_loader_source
        self.__data: pd.DataFrame = self.__data_loader.load_data_nyse()
        self.__log_returns: pd.DataFrame = pd.DataFrame()
        self.__ticker_columns = self.__data.columns

    def conduct_log_transformations_on_prices(
            self,
            is_corr_exclusionary=True
    ) -> tuple[pd.DataFrame, pd.Series]:
        log_prices: pd.DataFrame = pd.DataFrame()
        log_prices.index = pd.to_datetime(self.__data.index)

        for ticker in self.__ticker_columns:
            results: np.ndarray = np.log(self.__data[ticker])
            log_prices.insert(loc=0, column=ticker, value=results)

        corr_stack = log_prices.corr().stack()

        if is_corr_exclusionary:
            high_corr_stack_criterion = (corr_stack.values > 0.7) & (
                corr_stack.values != 1
            )
            high_corr_stack = corr_stack[high_corr_stack_criterion].astype(float)
            high_corr_stack = pd.DataFrame(high_corr_stack)
        else:
            diagonal_remover = corr_stack.values != 1
            high_corr_stack = corr_stack[diagonal_remover].astype(float)
            high_corr_stack = pd.DataFrame(high_corr_stack)

        high_corr_stack.columns = ["correlation"]
        return log_prices, high_corr_stack

    def compute_log_returns(self) -> pd.DataFrame:
        self.__log_returns.index = pd.to_datetime(self.__data.index)

        for ticker in self.__ticker_columns:
            results: np.ndarray = np.log(
                self.__data[ticker] / self.__data[ticker].shift(1)
            )
            self.__log_returns.insert(loc=0, column=ticker, value=results)

        # aimed to zero the first row and as failsafe since it is the first value and will be Nan
        self.__log_returns = self.__log_returns.fillna(0)
        return self.__log_returns

    def _engel_granger_fun(
            self,
            pairs : tuple[str, str],
            log_prices : pd.DataFrame,
    ) -> dict[str, float | str]:
        x = sm.add_constant(log_prices[pairs[1]])
        model = sm.OLS(log_prices[pairs[0]], x).fit()
        residual = model.resid
        adf = adfuller(residual)
        p = adf[1]
        c = model.params["const"]
        beta = model.params[pairs[1]]

        return {
            "direction" : f"{pairs[0]}~{pairs[1]}",
            "p" : p,
            "residual" : residual,
            "constant" : c,
            "hedge ratio" : beta,
            "t-statistic" : adf[0]
        }

    def _engel_granger_determinant(
            self,
            results_ab : dict[str, float | str],
            results_ba: dict[str, float | str],
    ) -> dict[str, float | str]:
        if results_ab["p"] <= results_ba["p"]:
            choice = results_ab
        else:
            choice = results_ba

        return choice

    def _MacKinnon_Critical_Value_formula(
            self,
            observations : int
    ) -> float:
        '''
        coefficients based from the source's Table 1: Response Surface of Critical Values
        https://www.econstor.eu/bitstream/10419/67744/1/616664753.pdf

        assumptions
        N -> INF as current observations per ticker is 1500+ days of closing prices
        no trend non stationary price nature of relative stock prices
        has constant as one stock is priced more / less than other
        acceptable error is 5%
        '''

        critical_value : float = -3.3377 + (-5.967 / observations) + (-8.98 / observations ** 2)
        return critical_value

    def _halflife_fun(
            self,
            is_cointegrated : bool,
            residuals : np.ndarray[Any, np.dtype[np.float64]],
    ) -> float:
        if is_cointegrated:
            residual_lag : np.ndarray[Any, np.dtype[np.float64]] = np.roll(residuals, 1)
            residual_lag[0] = 0
            residual_difference : np.ndarray[tuple[Any]]= residuals - residual_lag

            lagged : Any = sm.add_constant(residual_lag[1:])
            difference : np.ndarray[tuple[Any]]= residual_difference[1:]

            model : RegressionResults = sm.OLS(difference, lagged).fit()
            lambda_ = model.params[1]
            return -np.log(2) / lambda_

        return np.inf

    def engel_granger(
            self
    ) -> pd.DataFrame:
        log_prices, corr_stack = self.conduct_log_transformations_on_prices(False)
        crit_value = self._MacKinnon_Critical_Value_formula(self.__data.shape[0])

        p_residual : list[float] = []
        directions : list[str] = []
        hedge_ratio : list[float] = []
        constant : list[float] = []
        cointegrated : list[bool]= []
        t_statistic : list[float]= []
        half_life : list[float] = []

        for pairs in corr_stack.index:
            ab = self._engel_granger_fun((pairs[0], pairs[1]), log_prices)
            ba = self._engel_granger_fun((pairs[1], pairs[0]), log_prices)
            choice = self._engel_granger_determinant(ab, ba)
            is_cointegrated : bool = choice["t-statistic"] < crit_value
            resid : np.ndarray[Any, np.dtype[np.float64]] = np.array(choice["residual"])
            p_residual.append(choice["p"])
            directions.append(choice["direction"])
            hedge_ratio.append(choice["hedge ratio"])
            constant.append(choice["constant"])
            cointegrated.append(is_cointegrated)
            t_statistic.append(choice["t-statistic"])
            half_life.append(self._halflife_fun(is_cointegrated, resid))

        corr_stack.insert(loc=0, column="p", value=p_residual)
        corr_stack.insert(loc=0, column="direction", value=directions)
        corr_stack.insert(loc=0, column="constant", value=constant)
        corr_stack.insert(loc=0, column="hedge ratio", value=hedge_ratio)
        corr_stack.insert(loc=0, column="is cointegrated", value=cointegrated)
        corr_stack.insert(loc=0, column="t_statistic", value=t_statistic)
        corr_stack.insert(loc=0, column="half_life", value=half_life)

        return corr_stack
