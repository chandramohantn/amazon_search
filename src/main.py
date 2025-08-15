from query_processor.spell_correction import SpellCorrector

if __name__ == "__main__":
    spell_corrector = SpellCorrector()
    test_queries = [
        "niks shoos",
        "iphnoe",
        "apple iphone",
        "nike shoes",
        "niko shoos"
    ]

    corrected_results = {query: spell_corrector.spell_correction(query) for query in test_queries}
    print(corrected_results)