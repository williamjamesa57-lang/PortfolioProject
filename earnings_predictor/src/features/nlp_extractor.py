import re
from pathlib import Path

import bs4 as bs
from bs4 import BeautifulSoup, Tag
from numpy.matlib import empty
from yfinance import ticker
from edgar import Filing, use_local_storage, set_identity
from utils import data_loader
from utils.data_loader import DataLoader

import os

class NLPExtractor:
    def __init__(
            self,
            data_loader_source: data_loader.DataLoader,
            given_ticker : str
    ) -> None:
        self.__k_10_data : dict[str, str] | None= data_loader_source.load_data_sec_filings_ticker(ticker=given_ticker)
        self.__credentials : str = f"{os.getenv("SEC_EDGAR_USER_NAME")} {os.getenv("SEC_EDGAR_USER_EMAIL")}"

    def extract_features(
            self,
    ) -> list[str]:
        use_local_storage()
        set_identity(self.__credentials)

        keys : list[str] = list(self.__k_10_data.keys())
        extracted_features : list[str] = list()

        for key in keys:
            filing : tuple[str, str] = self.__k_10_data[key]
            risk_factors = ""

            if filing[0] == "html":
                risk_factors = self._extract_item_1a_from_html(filing[1])
                pass
            elif filing[0] == "sgml":
                item = self._load_via_sgml(filing[1])
                filings_obj = item.obj()
                risk_factors = filings_obj["Item 1A"]

            extracted_features.append(risk_factors)

        return extracted_features

    def _extract_item_1a_from_html(
            self,
            path : str
    ) -> str | None:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        soup = BeautifulSoup(text, "lxml")

        item_1a_heading = None
        for elem in soup.find_all(['b', 'strong', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'span']):
            text = elem.get_text(strip=True)
            if re.search(r'item\s+1a\.?', text, re.IGNORECASE):
                item_1a_heading = elem
                break

        if not item_1a_heading:
            return ""

        parts = [item_1a_heading.get_text(strip=True)]
        for sibling in item_1a_heading.find_next_siblings():
            sibling_text = sibling.get_text(strip=True)
            if sibling_text and re.search(r'item\s+\d+[a-z]?\.?', sibling_text, re.IGNORECASE):
                break
            parts.append(sibling.get_text(separator=' ', strip=True))

        return ' '.join(parts)

    def _load_via_sgml(
            self,
            target_file : str
    ) -> Filing:
        path = Path(target_file)
        return Filing.from_sgml(path)


if __name__ == "__main__":
    loader = DataLoader()
    nlp = NLPExtractor(loader, "AAPL")
    results = nlp.extract_features()
    pass