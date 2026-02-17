import datetime
import os
import re
from pathlib import Path
from typing import Any

from edgar.entity import EntityFilings
from scipy.constants import year
from sec_edgar_downloader import Downloader
from dotenv import load_dotenv
from edgar import Filing, use_local_storage, set_identity, Company
from datetime import datetime
import yfinance as yf
import numpy as np
import pandas as pd

import glob


class DataLoader:
    def __init__(self, data_dir=None):

        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir).resolve()

        # top 50 NYSE stock tickers with the most trade volume as of: 02/02/2026
        self.__tickers_nyse = [
            "VZ",
            "T",
            "F",
            "PFE",
            "FCX",
            "BAC",
            "XOM",
            "KO",
            "BEN",
            "WMT",
            "HPQ",
            "NEM",
            "MRK",
            "WFC",
            "DIS",
            "ABT",
            "SLB",
            "BMY",
            "NKE",
            "CL",
            "HAL",
            "GLW",
            "MO",
            "PG",
            "PEP",
            "C",
            "UNH",
            "JPM",
            "OXY",
            "DVN",
            "V",
            "JNJ",
            "KMB",
            "USB",
            "GIS",
            "LVS",
            "PM",
            "D",
            "GM",
            "MS",
            "GE",
            "BAX",
            "RTX",
            "UPS",
            "BA",
            "EOG",
            "HON",
            "EXC",
            "TGT",
        ]

        self.__tickers_sp_500 = [
            "NVDA",
            "AAPL",
            "MSFT",
            "AMZN",
            "GOOGL",
            "GOOG",
            "META",
            "TSLA",
            "AVGO",
            "AMD",
            "WMT",
            "LLY",
            "V",
            "XOM",
            "JNJ",
            "MU",
            "ORCL",
            "COST",
            "MA",
            "PG"
        ]

    def source_data_nyse(self):

        start_and_end_date = np.array(["2020-01-01", "2026-01-31"])

        datas_nyse = yf.download(
            tickers=self.__tickers_nyse,
            start=start_and_end_date[0],
            end=start_and_end_date[1],
            interval="1d",
            progress=False,
            auto_adjust=False,
            actions=False,
        )

        datas_nyse = datas_nyse["Adj Close"]

        if not isinstance(datas_nyse.index, pd.DatetimeIndex):
            datas_nyse.index = pd.to_datetime(datas_nyse.index)

        datas_nyse = datas_nyse.dropna(axis=1, how="all")
        datas_nyse = datas_nyse.ffill().bfill()

        datas_nyse.to_csv(self.data_dir / "nyse_50_stocks.csv")
        return datas_nyse

    def load_data_nyse(self):

        try:
            datas = pd.read_csv(self.data_dir / "nyse_50_stocks.csv")
            # checker if there is really 50 tickers in the data
            tickers_row = datas.iloc[0]
            tickers = set(tickers_row.dropna())
            assert len(tickers) == 50, f"Expected == 50 tickers got {len(tickers)}"
            # checker of observations if greater than 1000
            assert (
                datas.shape[0] > 1000
            ), f"Expected >1000 trading days got {datas.shape[0]}"
            datas["Date"] = pd.to_datetime(datas["Date"])
            datas.set_index("Date", inplace=True)
            return datas

        except FileNotFoundError:
            self.source_data_nyse()
            return self.load_data_nyse()

    def source_data_sec_filings(self):
        try:
            load_dotenv()
            full_name = os.getenv("SEC_EDGAR_USER_NAME")
            email = os.getenv("SEC_EDGAR_USER_EMAIL")
            assert full_name is not None, f"'SEC_EDGAR_USER_NAME' is not set"
            assert email is not None, f"'SEC_EDGAR_USER_EMAIL' is not set"

            downloader = Downloader(
                full_name,
                email,
                self.data_dir
            )

            for ticker in self.__tickers_sp_500:
                downloader.get(
                    "10-K",
                    ticker,
                    after=datetime.date(2005,12,1)
                )

            return downloader
        except FileNotFoundError:
            print(
                "[ERROR]: The .env file was not found. Please create one at root directory of the project."
            )

    def source_data_sec_filings_fragment(self, ticker):
        try:
            load_dotenv()
            full_name = os.getenv("SEC_EDGAR_USER_NAME")
            email = os.getenv("SEC_EDGAR_USER_EMAIL")
            assert full_name is not None, f"'SEC_EDGAR_USER_NAME' is not set"
            assert email is not None, f"'SEC_EDGAR_USER_EMAIL' is not set"
            downloader = Downloader(full_name, email, "./data/temp/")
            downloader.get("10-K", ticker)
            return downloader
        except FileNotFoundError:
            print(
                "[ERROR]: The .env file was not found. Please create one at root directory of the project."
            )

    def load_data_sec_filings(self) -> dict[str, str]:
        try:
            pattern = str(self.data_dir / "sec-edgar-filings/**/*.txt")
            fillings_dict = {}
            txt_files = glob.glob(
                pattern,
                recursive=True
            )

            for file_paths in txt_files:
                p = Path(file_paths)
                ticker = p.parts[7]
                numericals = p.parts[9]
                identification = f"{ticker}_{numericals}"

                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    fillings_dict[identification] = content

        except FileNotFoundError:
            self.source_data_sec_filings()
            return self.load_data_sec_filings()

        finally:
            return fillings_dict

    def load_data_sec_filings_ticker(self, ticker : str) -> dict[str, tuple[str, str]] | None:
        try:
            files : str = "sec-edgar-filings/" + ticker + "/**/*.txt"
            pattern : str = str(self.data_dir / files)
            fillings_dict : dict[str, tuple[str, str]] = {}
            txt_files = glob.glob(
                pattern,
                recursive=True
            )

            for file_paths in txt_files:
                p = Path(file_paths)
                ticker = p.parts[7]
                numericals = p.parts[9]
                identification = f"{ticker}_{numericals}"
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    head = f.read(2000)

                if re.search(r"<(html|!DOCTYPE\s+html)",head):
                    tag = "html"
                else:
                    tag = "sgml"

                fillings_dict[identification] = (
                    tag,
                    str(p)
                )

            return fillings_dict

        except FileNotFoundError:
            print(f"{ticker} not found on local data.")

    def load_data_sec_filings_ticker_edgar_tools(
            self,
            ticker : str
    ) -> EntityFilings | None:
        try:
            load_dotenv()
            identification : str = f"{os.getenv("SEC_EDGAR_USER_NAME")} {os.getenv("SEC_EDGAR_USER_EMAIL")}"
            set_identity(identification)
            company : Company = Company(ticker)
            years : list[int] = list(np.arange(2006, 2026, 1))
            tenk_filings = company.get_filings(form="10-K", year=years)
            return tenk_filings
        except Exception as e:
            print(f"error {e}")
