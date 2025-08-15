import numpy as np
from typing import Callable, Dict


class StringDistance:
    def __init__(self, metric: str = "levenshtein"):
        self.metrics: Dict[str, Callable[[str, str], float]] = {}
        self.register_builtin_metrics()
        if metric not in self.metrics:
            raise ValueError(f"Unknown metric '{metric}'. Available: {list(self.metrics.keys())}")
        self.metric_name = metric

    def register_builtin_metrics(self):
        self.metrics["levenshtein"] = self._levenshtein
        self.metrics["jaccard"] = self._jaccard
        self.metrics["cosine"] = self._cosine

    def set_metric(self, metric: str):
        if metric not in self.metrics:
            raise ValueError(f"Unknown metric '{metric}'. Available: {list(self.metrics.keys())}")
        self.metric_name = metric

    def compute(self, s1: str, s2: str) -> float:
        return self.metrics[self.metric_name](s1, s2)

    def _levenshtein(self, s1: str, s2: str) -> float:
        m, n = len(s1), len(s2)
        dp = np.zeros((m+1, n+1), dtype=int)
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
        return float(dp[m][n])

    def _jaccard(self, s1: str, s2: str) -> float:
        set1, set2 = set(s1), set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return 1 - intersection / union if union != 0 else 0

    def _cosine(self, s1: str, s2: str) -> float:
        from collections import Counter
        counter1, counter2 = Counter(s1), Counter(s2)
        all_chars = list(set(counter1) | set(counter2))
        v1 = np.array([counter1.get(ch, 0) for ch in all_chars])
        v2 = np.array([counter2.get(ch, 0) for ch in all_chars])
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return 1 - (dot / norm if norm != 0 else 0)

    def register_metric(self, name: str, func: Callable[[str, str], float]):
        self.metrics[name] = func


if __name__ == "__main__":
    dist = StringDistance(metric="levenshtein")
    print("Levenshtein:", dist.compute("kitten", "sitting"))

    dist.set_metric("jaccard")
    print("Jaccard:", dist.compute("night", "nacht"))

    dist.set_metric("cosine")
    print("Cosine:", dist.compute("night", "nacht"))
