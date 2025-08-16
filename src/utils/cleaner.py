import re

class Cleaner:
    def __init__(self):
        self.unwanted_symbols_pattern = re.compile(r'[^\\w\s\d]')

    def clean_query(self, query: str) -> str:
        query = re.sub(r"[^a-z0-9\s]", " ", query)
        query = re.sub(r"\s+", " ", query).strip()
        return query