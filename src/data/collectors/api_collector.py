"""API-based data collector for external data sources."""

import logging
from typing import Any, Iterator, Optional

import requests

logger = logging.getLogger(__name__)


class APICollector:
    """Collect training data from REST APIs with pagination support."""

    def __init__(
        self,
        base_url: str,
        headers: Optional[dict[str, str]] = None,
        timeout: int = 30,
        page_size: int = 100,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self.page_size = page_size

    def _request(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        method: str = "GET",
    ) -> Optional[dict[str, Any]]:
        """Execute API request and return JSON response."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            if method.upper() == "GET":
                resp = requests.get(url, params=params, headers=self.headers, timeout=self.timeout)
            else:
                resp = requests.post(url, json=params, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("API request failed %s: %s", url, e)
            return None

    def collect(
        self,
        endpoint: str = "",
        text_field: str = "text",
        id_field: str = "id",
        pagination_key: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[dict[str, Any]]:
        """Collect documents from API, optionally with pagination."""
        params: dict[str, Any] = {"limit": self.page_size}
        count = 0
        while True:
            data = self._request(endpoint, params=params)
            if not data:
                break
            items = data if isinstance(data, list) else data.get("items", data.get("results", []))
            if not items:
                break
            for item in items:
                if isinstance(item, dict) and text_field in item:
                    yield {
                        "id": str(item.get(id_field, count)),
                        "text": item[text_field],
                        "source": self.base_url,
                        "source_type": "api",
                        "raw": item,
                    }
                    count += 1
                    if limit is not None and count >= limit:
                        return
            if not pagination_key or pagination_key not in data:
                break
            params = data.get("next_params", {}) or params
            if "offset" in params:
                params["offset"] += len(items)
            elif "page" in params:
                params["page"] += 1
            else:
                break
