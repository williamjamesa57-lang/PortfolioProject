import bs4
import sec_parser as sp
import bs4 as bs

from utils import data_loader
from utils.data_loader import DataLoader

class NLPExtractor:
    def __init__(self, data_loader_source: data_loader.DataLoader) -> None:
        self.__k_10_data = data_loader_source.load_data_sec_filings()

    def extract_features(self) -> list[str]:
        sample = self.__k_10_data.values
        texts = sample
        tags = []#[tag for tag in sample[0].children if isinstance(tag, bs4.Tag)]
        return tags
