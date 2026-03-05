"""Text cleaning and normalization for LLM training data."""

import re
import unicodedata
from typing import Optional

class TextCleaner:
    """Clean and normalize raw text for training."""

    def __init__(
        self,
        min_length: int = 10,
        max_length: Optional[int] = None,
        normalize_unicode: bool = True,
        remove_urls: bool = True,
        remove_extra_whitespace: bool = True,
        strip_html: bool = True,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.normalize_unicode = normalize_unicode
        self.remove_urls = remove_urls
        self.remove_extra_whitespace = remove_extra_whitespace
        self.strip_html = strip_html
        self._url_pattern = re.compile(
            r"https?://[^\s]+|www\.[^\s]+",
            re.IGNORECASE,
        )
        self._html_pattern = re.compile(r"<[^>]+>")

    def clean(self, text: str) -> str:
        """Apply cleaning pipeline to text."""
        if not text or not isinstance(text, str):
            return ""
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        if self.strip_html:
            text = self._html_pattern.sub(" ", text)
        if self.remove_urls:
            text = self._url_pattern.sub(" ", text)
        if self.remove_extra_whitespace:
            text = re.sub(r"\s+", " ", text).strip()
        if self.min_length and len(text) < self.min_length:
            return ""
        if self.max_length and len(text) > self.max_length:
            text = text[: self.max_length].rsplit(" ", 1)[0]
        return text.strip()

    def clean_document(self, doc: dict) -> Optional[dict]:
        """Clean text field in document; drop if empty."""
        text_key = "text" if "text" in doc else "content"
        if text_key not in doc:
            return None
        cleaned = self.clean(str(doc[text_key]))
        if not cleaned:
            return None
        out = dict(doc)
        out[text_key] = cleaned
        return out
