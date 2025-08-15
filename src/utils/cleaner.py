import re

class Cleaner:
    def __init__(self):
        self.unwanted_symbols_pattern = re.compile(r'[^\\w\s\d]')

    def clean_query(self, query: str) -> str:
        query = query.lower()
        query = query.strip()
        # query = self.unwanted_symbols_pattern.sub('', query)
        return query