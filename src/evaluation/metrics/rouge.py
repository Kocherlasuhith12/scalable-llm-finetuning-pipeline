"""ROUGE metric for text generation evaluation."""

from typing import Optional

try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False


def compute_rouge(
    predictions: list[str],
    references: list[str],
    types: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L (or specified types)."""
    if not _ROUGE_AVAILABLE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    types = types or ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(types, use_stemmer=True)
    totals = {t: {"f": 0.0, "n": 0} for t in types}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for t in types:
            totals[t]["f"] += scores[t].fmeasure
            totals[t]["n"] += 1
    return {t: totals[t]["f"] / totals[t]["n"] if totals[t]["n"] else 0.0 for t in types}
