"""Deduplication for training data."""

import hashlib
import logging
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


class Deduplicator:
    """Exact and fuzzy deduplication of documents."""

    def __init__(
        self,
        threshold: float = 0.9,
        use_hash: bool = True,
        text_key: str = "text",
    ) -> None:
        self.threshold = threshold
        self.use_hash = use_hash
        self.text_key = text_key
        self._seen_hashes: set[str] = set()
        self._min_hash_size = 64  # simhash or first N chars for quick check

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _normalize(self, text: str) -> str:
        return " ".join(text.split()).strip().lower()

    def is_duplicate(self, doc: dict) -> bool:
        """Check if document is duplicate (exact hash)."""
        text = doc.get(self.text_key, doc.get("content", ""))
        if not text:
            return True
        h = self._hash_text(self._normalize(text))
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False

    def dedupe_stream(
        self,
        documents: Iterator[dict],
        yield_duplicates: bool = False,
    ) -> Iterator[dict]:
        """Deduplicate a stream of documents by content hash."""
        for doc in documents:
            dup = self.is_duplicate(doc)
            if not dup:
                yield doc
            elif yield_duplicates:
                yield {**doc, "_duplicate": True}

    def reset(self) -> None:
        """Clear seen hashes for new run."""
        self._seen_hashes.clear()
