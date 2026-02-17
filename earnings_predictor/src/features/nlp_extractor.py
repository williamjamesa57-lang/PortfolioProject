from typing import Any
import numpy as np
import pandas as pd
from edgar.entity import EntityFilings
from utils import data_loader
from sklearn.feature_extraction.text import TfidfVectorizer

import os

class NLPExtractor:
    def __init__(
            self,
            data_loader_source: data_loader.DataLoader,
    ) -> None:
        self.__data_loader_source = data_loader_source
        self.__credentials : str = f"{os.getenv("SEC_EDGAR_USER_NAME")} {os.getenv("SEC_EDGAR_USER_EMAIL")}"

    def extract_features_from_edgar_tools(
            self,
            ticker : str
    ) -> list[dict[str, Any]]:
        tenk_filings : EntityFilings = self.__data_loader_source.load_data_sec_filings_ticker_edgar_tools(ticker)
        risk_data : list[dict[str, Any]] = []
        for entry in tenk_filings:
            try:
                form_objects = entry.obj()
                risk_text = form_objects.risk_factors
                risk_data.append({
                    "filing_date" : entry.filing_date,
                    "report_date" : entry.report_date,
                    "accession_number" : entry.accession_number,
                    "risk_factor" : risk_text
                })

            except Exception as e:
                print(f"error {e}")

        return risk_data

    def get_top_n_words(
            self,
            n : int,
            extracted_features : list[dict[str, Any]]
    )-> pd.DataFrame:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            token_pattern= r"(?u)\b[A-Za-z]{3,}\b",
            max_df=0.8,
            min_df=2,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
        )

        risk_factor_list : list[str] = [item.get("risk_factor")
                                        if item.get("risk_factor") is not None
                                        else " "
                                        for item in extracted_features]
        tfidf_matrix = vectorizer.fit_transform(risk_factor_list)
        feature_names = vectorizer.get_feature_names_out()
        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

        word_scoring = list(zip(feature_names, mean_tfidf))
        word_scoring.sort(key=lambda x: x[1], reverse=True)

        return pd.DataFrame(word_scoring, columns=["word", "tfdif score"])