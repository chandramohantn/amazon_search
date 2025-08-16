import os
from typing import List, Optional
from utils.load_cache import LoadCache
from utils.distances import StringDistance
from dotenv import load_dotenv

load_dotenv()

class SpellCorrector:
    def __init__(self):
        self.top_k_candidates = int(os.getenv('TOP_K_CANDIDATES', 5))
        self.string_metric = os.getenv('STRING_DISTANCE_METRIC', 'levenshtein')
        self.count_oov_tokens = int(os.getenv('OOV_TOKENS_COUNT', 1))
        self.cache_loader = LoadCache()
        self.cache_data = self.cache_loader.load_all_cache()
        self.distance_calculator = StringDistance()
    
    def is_oov(self, word: str) -> bool:
        """
        Check if the word is out of vocabulary (OOV).
        Args:
            word (str): Input word to check.
        Returns:
            bool: True if the word is OOV, False otherwise.
        """
        return word not in self.cache_data['vocab']

    def get_closest_products(self, query: str) -> List[str]:
        """
        Get the closest products to the query based on the specified string distance metric.
        Args:
            query (str): Input query to find closest products for.
        Returns:
            List[str]: List of closest products to the query.
        """
        self.distance_calculator.set_metric(self.string_metric)
        vocabulary = self.cache_data.get('vocab', [])
        if not vocabulary:
            print("Vocabulary is empty or not loaded.")
            return []

        distances = [(product, self.distance_calculator.compute(query, product)) for product in vocabulary]
        distances.sort(key=lambda x: x[1])
        return [product for product, _ in distances[:self.top_k_candidates]]

    def score_candidates(self, candidates: List[str]) -> Optional[str]:
        """
        Score the candidates based on their frequency in the product queries.
        This is a simple tie-breaker that selects the most frequent candidate.
        Args:
            candidates (List[str]): List of candidate products to score.

        Returns:
            Optional[str]: The candidate with the highest frequency or None if no candidates are provided.
        """
        if not candidates:
            print("No candidates provided for scoring.")
            return None
        # Tie-break using product queries (frequency)
        return max(candidates, key=lambda x: self.cache_data['product_queries'].get(x, 0))
    
    def check_confidence(self, query: str) -> bool:
        """
        Check the confidence of the spell correction.
        Args:
            query (str): Input query to check confidence for.
        Returns:
            bool: True if the query is confident, False otherwise.
        """
        count_oov = sum(1 for token in query.split() if self.is_oov(token))
        if count_oov > self.count_oov_tokens:
            print(f"Query '{query}' has {count_oov} out-of-vocabulary tokens.")
            return False
        print(f"Query '{query}' is confident with no OOV tokens.")
        return True

    def apply_distance_based_correction(self, query: str) -> Optional[str]:
        """
        Apply distance-based correction to the query.
        Args:
            query (str): Input query to correct.
        Returns:
            Optional[List[str]]: List of corrected queries or None if no candidates found.
        """
        tokens = query.split()
        corrected_tokens = []
        for token in tokens:
            if not self.is_oov(token):
                corrected_tokens.append(token)
                continue
        
            candidates = self.get_closest_products(token)
            if not candidates:
                print(f"No candidates found for query '{token}'.")
                corrected_tokens.append(token)
            elif len(candidates) == 1:
                corrected_tokens.append(candidates[0])
            else:
                scored_candidate = self.score_candidates(candidates)
                if scored_candidate:
                    corrected_tokens.append(scored_candidate)
                else:
                    print(f"No scored candidates found for query '{token}'.")
                    corrected_tokens.append(token)
            
        return " ".join(corrected_tokens)

    def apply_neural_correction(self, query: str) -> Optional[str]:
        """
        Apply neural correction to the query.
        This is a placeholder for future implementation.
        Args:
            query (str): Input query to correct.
        Returns:
            Optional[str]: Corrected query or None if not implemented.
        """
        print("Neural correction is not implemented yet.")
        pass

    def apply_bert_correction(self, query: str) -> Optional[str]:
        """
        Apply BERT-based correction to the query.
        This is a placeholder for future implementation.
        Args:
            query (str): Input query to correct.
        Returns:
            Optional[str]: Corrected query or None if not implemented.
        """
        print("BERT correction is not implemented yet.")
        pass

    def spell_correction(self, query: str) -> Optional[str]:
        """
        Apply the best available correction method to the query.
        Args:
            query (str): Input query to correct.
        Returns:
            Optional[str]: Corrected query or None if no correction is applied.
        """
        print(f"\nApplying correction for query: {query}")

        # 1. Check Product Spell Cache
        if query in self.cache_data['product_spell']:
            print("✔ Using cached correction.")
            return self.cache_data['product_spell'][query]
        
        # 2. Check for OOV
        tokens = query.split()
        if all(not self.is_oov(token) for token in tokens):
            print("✔ Query is valid, no correction needed.")
            return query

        # 3. Apply Distance Based Correction
        print("→ Applying Distance Based Correction...")
        corrected_query = self.apply_distance_based_correction(query)

        # 4. Check Confidence of Distance Based Correction
        if self.check_confidence(corrected_query):
            print("⚠️ Low confidence in distance based correction. Escalating to Neural Correction...")
            corrected_query = self.apply_neural_correction(query)
            if corrected_query is None:
                return None

            if self.check_confidence(corrected_query):
                print("⚠️ Still low confidence. Escalating to BERT Correction...")
                corrected_query = self.apply_bert_correction(query)
        
        # 5. Return final corrected query
        print(f"✅ Final Corrected Query: {corrected_query}")
        return corrected_query
