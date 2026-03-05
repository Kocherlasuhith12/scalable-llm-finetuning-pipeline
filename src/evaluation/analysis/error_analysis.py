"""Error analysis for model predictions."""

import logging
from collections import defaultdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Analyze failure modes and error patterns."""

    def __init__(self, group_by: Optional[list[str]] = None) -> None:
        self.group_by = group_by or []
        self.errors: list[dict[str, Any]] = []

    def add(self, prediction: str, reference: str, metadata: Optional[dict] = None) -> None:
        self.errors.append({
            "prediction": prediction,
            "reference": reference,
            "metadata": metadata or {},
        })

    def add_correct(self, prediction: str, reference: str, metadata: Optional[dict] = None) -> None:
        pass  # optional: track correct for balance

    def by_length(self, pred_key: str = "prediction", ref_key: str = "reference") -> dict[str, int]:
        """Bucket errors by reference length."""
        buckets = defaultdict(int)
        for e in self.errors:
            ref_len = len(e[ref_key])
            bucket = "short" if ref_len < 50 else "medium" if ref_len < 200 else "long"
            buckets[bucket] += 1
        return dict(buckets)

    def summary(self) -> dict[str, Any]:
        return {
            "total_errors": len(self.errors),
            "by_length": self.by_length(),
        }
