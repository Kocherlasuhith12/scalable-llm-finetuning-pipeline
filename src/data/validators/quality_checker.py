"""Quality scoring and filtering for training data."""

import re
from typing import Optional

class QualityChecker:
    """Score and filter documents by quality heuristics."""

    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 100_000,
        min_alpha_ratio: float = 0.5,
        max_repeat_ratio: float = 0.3,
        min_unique_ratio: float = 0.5,
        blocklist_patterns: Optional[list[str]] = None,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.min_alpha_ratio = min_alpha_ratio
        self.max_repeat_ratio = max_repeat_ratio
        self.min_unique_ratio = min_unique_ratio
        self.blocklist = [re.compile(p, re.I) for p in (blocklist_patterns or [])]

    def _alpha_ratio(self, text: str) -> float:
        alpha = sum(1 for c in text if c.isalpha())
        return alpha / len(text) if text else 0.0

    def _repeat_ratio(self, text: str, window: int = 10) -> float:
        if len(text) < window:
            return 0.0
        chunks = [text[i : i + window] for i in range(0, len(text) - window + 1, window)]
        unique = len(set(chunks))
        return 1.0 - (unique / len(chunks)) if chunks else 0.0

    def _unique_ratio(self, text: str) -> float:
        words = text.split()
        return len(set(words)) / len(words) if words else 0.0

    def score(self, text: str) -> float:
        """Return quality score in [0, 1]. Higher is better."""
        if not text or len(text) < self.min_length or len(text) > self.max_length:
            return 0.0
        for pat in self.blocklist:
            if pat.search(text):
                return 0.0
        alpha = self._alpha_ratio(text)
        if alpha < self.min_alpha_ratio:
            return 0.0
        repeat = self._repeat_ratio(text)
        if repeat > self.max_repeat_ratio:
            return 0.0
        unique = self._unique_ratio(text)
        if unique < self.min_unique_ratio:
            return 0.0
        return min(1.0, alpha * 0.4 + (1 - repeat) * 0.3 + unique * 0.3)

    def filter_document(self, doc: dict, min_score: float = 0.5, text_key: str = "text") -> Optional[dict]:
        """Return doc with quality score if above threshold, else None."""
        text = doc.get(text_key, doc.get("content", ""))
        s = self.score(str(text))
        if s < min_score:
            return None
        out = dict(doc)
        out["quality_score"] = s
        return out
