"""BLEU metric for text generation evaluation."""

from typing import Optional

try:
    import sacrebleu
    _SACREBLEU_AVAILABLE = True
except ImportError:
    _SACREBLEU_AVAILABLE = False


def compute_bleu(
    predictions: list[str],
    references: list[list[str]],
    smooth_method: str = "exp",
) -> dict[str, float]:
    """Compute BLEU score (sacrebleu). references can be list of refs per prediction."""
    if not _SACREBLEU_AVAILABLE:
        return {"bleu": 0.0}
    if not references:
        return {"bleu": 0.0}
    # Flatten to one ref per pred if list of strings
    refs = [[r] if isinstance(r, str) else r for r in references]
    while len(refs) < len(predictions):
        refs.append([""])
    refs = refs[: len(predictions)]
    bleu = sacrebleu.corpus_bleu(predictions, refs, smooth_method=smooth_method)
    return {"bleu": bleu.score / 100.0}
