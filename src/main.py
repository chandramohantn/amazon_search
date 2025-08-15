from query_processor.spell_correction import SpellCorrector
from utils.cleaner import Cleaner

if __name__ == "__main__":
    cleaner = Cleaner()
    spell_corrector = SpellCorrector()
    test_queries = [
        "niks shoos",
        "iphnoe",
        "apple iphone",
        "nike shoes",
        "niko shoos"
    ]

    for query in test_queries:
        corrected_result = spell_corrector.spell_correction(cleaner.clean_query(query))
        print(f"Corrected query: {corrected_result}\n")