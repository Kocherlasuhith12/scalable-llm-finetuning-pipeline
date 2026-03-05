"""Custom task-specific metrics."""

import re
from typing import Any, Callable, Optional


def exact_match(predictions: list[str], references: list[str], normalize: bool = True) -> float:
    """Exact match accuracy."""
    if normalize:
        preds = [p.strip().lower() for p in predictions]
        refs = [r.strip().lower() for r in references]
    else:
        preds, refs = predictions, references
    return sum(p == r for p, r in zip(preds, refs)) / len(preds) if preds else 0.0


def prefix_match(predictions: list[str], references: list[str], strip: bool = True) -> float:
    """Reference is prefix of prediction (e.g. completion)."""
    if strip:
        preds = [p.strip() for p in predictions]
        refs = [r.strip() for r in references]
    else:
        preds, refs = predictions, references
    return sum(ref in p for p, ref in zip(preds, refs)) / len(preds) if preds else 0.0


def compute_custom_metrics(
    predictions: list[str],
    references: list[str],
    metrics: Optional[list[tuple[str, Callable]]] = None,
) -> dict[str, float]:
    """Compute a set of custom metrics."""
    metrics = metrics or [
        ("exact_match", lambda p, r: exact_match(p, r)),
        ("prefix_match", lambda p, r: prefix_match(p, r)),
    ]
    return {name: fn(predictions, references) for name, fn in metrics}
