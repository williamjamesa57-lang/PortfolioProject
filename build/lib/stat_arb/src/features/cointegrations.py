from statsmodels.tsa.stattools import coint, adfuller
from utils import data_loader

import statsmodels.api as sm
import pandas as pd
import numpy as np


def determine_top_cointegrated_pairs(given_stack: pd.DataFrame, n: int) -> pd.DataFrame:
    given_stack.sort_values(by=["p"], inplace=True)
    corr_stack_s = given_stack.iloc[0:n]
    return corr_stack_s


class CointegrationEngine:
    def __init__(self, data_loader_source: data_loader.DataLoader) -> None:
        self.__data_loader: data_loader.DataLoader = data_loader_source
        self.__data: pd.DataFrame = self.__data_loader.load_data_nyse()
        self.__log_returns: pd.DataFrame = pd.DataFrame()
        self.__ticker_columns = self.__data.columns

    def conduct_log_transformations_on_prices(self, is_corr_exclusionary=True):
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

    def engel_granger(self) -> pd.DataFrame:
        log_prices, corr_stack = self.conduct_log_transformations_on_prices(False)

        p_residual = []
        directions = []
        hedge_ratio = []
        constant = []

        for pairs in corr_stack.index:
            tickers = [pairs[0], pairs[1]]

            # A~B
            x_ab = sm.add_constant(log_prices[tickers[1]])
            model_ab = sm.OLS(log_prices[tickers[0]], x_ab).fit()
            residual_ab = model_ab.resid
            adf_ab = adfuller(residual_ab)
            p_ab = adf_ab[1]

            # B~A
            x_ba = sm.add_constant(log_prices[tickers[0]])
            model_ba = sm.OLS(log_prices[tickers[1]], x_ba).fit()
            residual_ba = model_ba.resid
            adf_ba = adfuller(residual_ba)
            p_ba = adf_ba[1]

            if p_ab <= p_ba:
                p_residual.append(p_ab)
                directions.append(f"{pairs[0]}{pairs[1]}")
                constant.append(model_ab.params["const"])
                hedge_ratio.append(model_ab.params[pairs[1]])
            else:
                p_residual.append(p_ba)
                directions.append(f"{pairs[1]}{pairs[0]}")
                constant.append(model_ba.params["const"])
                hedge_ratio.append(model_ba.params[pairs[0]])

        corr_stack.insert(loc=0, column="p", value=p_residual)
        corr_stack.insert(loc=0, column="direction", value=directions)
        corr_stack.insert(loc=0, column="constant", value=constant)
        corr_stack.insert(loc=0, column="hedge ratio", value=hedge_ratio)

        return corr_stack


if __name__ == "__main__":
    cointegration = CointegrationEngine(data_loader_source=data_loader.DataLoader())
    corr_stack = cointegration.engel_granger()
    top_5_corr_stack = determine_top_cointegrated_pairs(corr_stack, n=5)
    top_5_corr_stack.to_csv("top_5_corr_stack.csv")
    pass
