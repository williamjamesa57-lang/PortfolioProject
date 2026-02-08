import pandas as pd
import pytest

from pathlib import Path
from utils.data_loader import DataLoader

DATA_DIR = Path(__file__).parent.parent / "data"

class TestUnitNYSEData:

    def setup_method(self):
        self._loader = DataLoader()
        self._datas = self._loader.load_data_nyse()

    def test_load_nyse_stocks_data_frame(self):
        assert isinstance(self._datas, pd.DataFrame), \
            f"Data is not data frame instead: {type(self._datas)}"
        assert not self._datas.empty, \
            f"Data is empty!!"

    def test_load_nyse_stocks_has_correct_tickers(self):
        expectations = {
            "VZ","T","F","PFE","FCX","BAC","XOM","KO","BEN","WMT",
            "HPQ","NEM","MRK","WFC","DIS","ABT","SLB","BMY","NKE",
            "CL","HAL","GLW","MO","PG","PEP","C","UNH","JPM","OXY",
            "DVN","V","JNJ","KMB","USB","GIS","LVS","PM","D","GM",
            "MS","GE","BAX","RTX","UPS","BA","EOG","HON","EXC","TGT"
        }

        assert expectations.issubset(set(self._datas)), \
            f"missing some tickers. Got: {self._datas.columns.tolist()}."

    def test_load_nyse_stocks_is_date_time(self):
        assert pd.api.types.is_datetime64_any_dtype(self._datas.index), \
            f"Index not of date time type instead {type(self._datas.index).__name__}"

    def test_all_prices_non_negative_real_value(self):
        assert (self._datas >= 0).all().all(), \
            f"not all prices are negative!!"

class TestUnitSECFilings:

    def setup_class(self):
        self._loader = DataLoader()
        self._datas = self._loader.load_data_sec_filings()

    def test_load_sec_filings_has_correct_tickers(self):
        expectations = {
            "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "AMD",
            "WMT", "LLY", "JPM", "V", "XOM", "JNJ", "MU", "MA", "ORCL", "COST"
        }

        loaded_tickers = {key.split("_")[0] for key in self._datas.keys()}
        missing = expectations - loaded_tickers

        assert not missing, f"missing tickers: {missing}"

    def test_sec_filings_data_integrity(self):
        assert isinstance(self._datas, dict), f"Data is not a dict!!"
        assert len(self._datas) > 0, f"Data is empty!!"

        sample_key, sample_value = next(iter(self._datas.items()))
        assert isinstance(sample_key, str), f"Key is not a string!!"
        assert isinstance(sample_value, str), f"Value is not a string!!"
        # covers the empty filing as String None is <5000
        assert len(sample_value) > 50000, f"Value is lower than expected!!"

        assert "SECURITIES AND EXCHANGE COMMISSION" in sample_value
        assert "UNITED STATES" in sample_value
        assert "FORM 10-K" in sample_value