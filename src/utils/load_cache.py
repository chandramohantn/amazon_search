import os
import json
from dotenv import load_dotenv

class LoadCache:
    def __init__(self):
        load_dotenv()
        self.cache_folder = os.getenv('CACHE_FOLDER', os.path.join(os.path.dirname(__file__), '../../cache'))

    def load_json(self, filename):
        file_path = os.path.join(self.cache_folder, filename)
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"File {filename} not found in cache folder.")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file {filename}.")
            return None

    def load_all_cache(self):
        return {
            'product_catalog': self.load_json('product_catalog.json')["inventory"],
            'product_queries': self.load_json('product_queries.json'),
            'product_spell': self.load_json('product_spell.json'),
            'vocab': self.load_json('vocab.json')["vocabulary"]
        }

