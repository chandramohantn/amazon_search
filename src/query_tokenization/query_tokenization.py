import math
from typing import List, Tuple
from utils.load_cache import LoadCache
from collections import defaultdict
from utils.cleaner import Cleaner


class QuerySegmenter:
    def __init__(self, vocab=None, unigram_prob=None, bigram_prob=None, neural_model=None):
        self.vocab = vocab if vocab else set()
        self.unigram_prob = unigram_prob if unigram_prob else defaultdict(float)
        self.bigram_prob = bigram_prob if bigram_prob else defaultdict(lambda: defaultdict(float))
        self.neural_model = neural_model

    # Segment the query using a greedy dictionary based approach
    def dictionary_segmentation(self, query: str) -> Tuple[List[str], List[float]]:
        """
        Segment the query using a dictionary-based approach.
        Args:
            query (str): Input query to segment.
        Returns:
            List[str]: List of segmented words.
        """
        words = query.split()
        segments = []
        matched_count = 0

        i = 0
        while i < len(words):
            match_found = False
            # Try longest match first (greedy)
            for j in range(len(words), i, -1):
                phrase = " ".join(words[i:j])
                if phrase in self.dictionary:
                    segments.append(phrase)
                    matched_count += 1
                    i = j
                    match_found = True
                    break
            if not match_found:
                segments.append(words[i])
                i += 1
        
        confidence = matched_count / len(segments)
        return segments, confidence

    def unigram_segmentation(self, query: str) -> Tuple[List[str], List[float]]:
        """
        Segment the query using a unigram-based approach.
        Args:
            query (str): Input query to segment.
        Returns:
            List[str]: List of segmented words.
        """
        chars = query.replace(" ", "")
        n = len(chars)
        best_seg = [""] * (n+1)
        best_score = [-math.inf] * (n+1)
        best_score[0] = 0

        for i in range(n):
            for j in range(i+1, n+1):
                token = chars[i:j]
                p_token = self.unigram_probs.get(token, 1e-6)
                score = best_score[i] + math.log(p_token)
                if score > best_score[j]:
                    best_score[j] = score
                    best_seg[j] = (best_seg[i] + " " + token).strip()

        tokens = best_seg[n].split()
        confidence = math.exp(best_score[n] / len(tokens))
        return tokens, confidence

    def bigram_segmentation(self, query: str) -> Tuple[List[str], List[float]]:
        """
        Segment the query using a bigram-based approach.
        Args:
            query (str): Input query to segment.
        Returns:
            List[str]: List of segmented words.
        """
        chars = query.replace(" ", "")
        n = len(chars)
        best_seg = [""] * (n+1)
        best_score = [-math.inf] * (n+1)
        best_score[0] = 0

        for i in range(n):
            for j in range(i+1, n+1):
                token = chars[i:j]
                if i > 0:
                    prev_token = best_seg[i].split()[-1]
                    p_token = self.bigram_prob.get(prev_token, {}).get(token, 1e-6)
                else:
                    p_token = self.unigram_prob.get(token, 1e-6)
                score = best_score[i] + math.log(p_token)
                if score > best_score[j]:
                    best_score[j] = score
                    best_seg[j] = (best_seg[i] + " " + token).strip()

        tokens = best_seg[n].split()
        confidence = math.exp(best_score[n] / len(tokens))
        return tokens, confidence
    
    def neural_segmentation(self, query):
        if not self.neural_model:
            return query.split(), 0.0
        tokens, conf = self.neural_model(query)
        return tokens, conf

    def segment(self, query):
        # Clean the query
        cleaner = Cleaner()
        query = cleaner.clean_query(query)

        # Try dictionary-based segmentation
        dict_tokens, dict_conf = self.dictionary_segmentation(query)
        if dict_conf > 0.8:
            return {"method": "dictionary", "tokens": dict_tokens, "confidence": dict_conf}

        # Fallback to unigram segmentation
        unigram_tokens, unigram_conf = self.unigram_segmentation(query)
        if unigram_conf > 0.7:
            return {"method": "unigram", "tokens": unigram_tokens, "confidence": unigram_conf}

        # Fallback to bigram segmentation
        bigram_tokens, bigram_conf = self.bigram_segmentation(query)
        if bigram_conf > 0.7:
            return {"method": "bigram", "tokens": bigram_tokens, "confidence": bigram_conf}

        # Final fallback to neural model
        neural_tokens, neural_conf = self.neural_segmentation(query)
        return {"method": "neural", "tokens": neural_tokens, "confidence": neural_conf}
