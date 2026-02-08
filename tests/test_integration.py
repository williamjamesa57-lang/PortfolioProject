import pytest
import pandas as pd

from pathlib import Path
from utils.data_loader import DataLoader

DIR_PATH = Path(__file__).parent.parent / "data"


@pytest.mark.slow
class TestIntegrationNYSEData:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self._loader = DataLoader()
        self._datas = self._loader.source_data_nyse()

    def test_download_integrity(self):
        assert not self._datas.empty, f"downloaded data is empty!!"
        # 0 indexed counting
        assert (
            self._datas.shape[1] == 49
        ), f"data not fully downloaded tickers count: {self._datas.shape[1]} not 49!!"
        assert pd.api.types.is_datetime64_any_dtype(
            self._datas.index
        ), f"index is not time-series data!!"
        assert (self._datas >= 0).all().all(), f"all values not positive real number!!!"


@pytest.mark.slow
class TestIntegrationSECK10Data:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self._loader = DataLoader()
        self._datas = self._loader.source_data_sec_filings_fragment("AAPL")

    def test_download_integrity(self):
        aapl_dir = DIR_PATH / "temp" / "sec-edgar-filings" / "AAPL"
        assert aapl_dir.exists(), f"{aapl_dir} not exist!!"
        txt_files = list(aapl_dir.rglob("full-submission.txt"))
        assert len(txt_files) > 0, f"{aapl_dir} empty!!"

        content = txt_files[0].read_text()
        assert len(content) > 50000, f"{content} relatively short for a 1-K0 document!!"
        assert "SECURITIES AND EXCHANGE COMMISSION" in content
        assert "Form 10-K" in content
        assert "UNITED STATES" in content
