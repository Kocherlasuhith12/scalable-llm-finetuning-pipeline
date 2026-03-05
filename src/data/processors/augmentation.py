"""Text augmentation for LLM training data."""

import random
import re
from typing import Callable, Optional

from .cleaner import TextCleaner


class TextAugmenter:
    """Apply augmentation techniques to increase data diversity."""

    def __init__(
        self,
        techniques: Optional[list[str]] = None,
        p: float = 0.3,
        cleaner: Optional[TextCleaner] = None,
    ) -> None:
        self.techniques = techniques or ["back_translation", "synonym", "noise"]
        self.p = p
        self.cleaner = cleaner or TextCleaner()

    def _add_typo(self, text: str) -> str:
        """Random character swap/delete for robustness."""
        if len(text) < 3:
            return text
        chars = list(text)
        i = random.randint(0, len(chars) - 1)
        if random.random() < 0.5 and len(chars) > 1:
            j = (i + 1) % len(chars)
            chars[i], chars[j] = chars[j], chars[i]
        else:
            chars[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
        return "".join(chars)

    def _add_whitespace_noise(self, text: str) -> str:
        """Add occasional extra spaces."""
        words = text.split()
        if not words:
            return text
        out = []
        for w in words:
            if random.random() < self.p:
                out.append(w + " ")
            out.append(w)
        return " ".join(out)

    def _sentence_shuffle(self, text: str) -> str:
        """Shuffle sentences (weak augmentation for diversity)."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 1:
            return text
        random.shuffle(sentences)
        return " ".join(sentences)

    def augment(self, text: str, technique: Optional[str] = None) -> str:
        """Apply one random augmentation with probability p."""
        if random.random() > self.p:
            return text
        tech = technique or random.choice(self.techniques)
        if "noise" in tech or "typo" in tech:
            return self._add_typo(text)
        if "whitespace" in tech:
            return self._add_whitespace_noise(text)
        if "shuffle" in tech:
            return self._sentence_shuffle(text)
        return text

    def augment_document(self, doc: dict, text_key: str = "text") -> dict:
        """Augment text in document and return new doc."""
        if text_key not in doc:
            return doc
        out = dict(doc)
        out[text_key] = self.augment(str(doc[text_key]))
        return out
