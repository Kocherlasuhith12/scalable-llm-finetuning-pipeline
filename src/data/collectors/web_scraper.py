"""Web scraper for collecting training data from web sources."""

import logging
from pathlib import Path
from typing import Any, Iterator, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebScraper:
    """Collect text data from web URLs with configurable extraction."""

    def __init__(
        self,
        user_agent: str = "LLM-Finetuning-Pipeline/1.0",
        timeout: int = 30,
        max_content_length: int = 1_000_000,
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.timeout = timeout
        self.max_content_length = max_content_length

    def fetch_url(self, url: str) -> Optional[str]:
        """Fetch and extract main text content from a URL."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            if len(response.content) > self.max_content_length:
                logger.warning("Content too large for %s, skipping", url)
                return None
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return text if text else None
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            return None

    def scrape_urls(
        self, urls: list[str], output_dir: Optional[Path] = None
    ) -> Iterator[dict[str, Any]]:
        """Scrape multiple URLs and yield document dicts."""
        output_dir = output_dir or Path(".")
        for i, url in enumerate(urls):
            text = self.fetch_url(url)
            if text:
                doc = {"id": f"web_{i}", "source": url, "text": text, "source_type": "web"}
                yield doc
                if output_dir:
                    out_path = output_dir / f"doc_{i}.txt"
                    out_path.write_text(text, encoding="utf-8")
