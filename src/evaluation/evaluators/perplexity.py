"""Perplexity evaluation for language models."""

import math
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader


def compute_perplexity(
    model: Any,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> float:
    """Compute perplexity (exp(cross-entropy)) on a dataset."""
    model.eval()
    device = device or next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            if max_batches is not None and n >= max_batches:
                break
            inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
            labels = batch.get("labels")
            if labels is not None:
                inputs["labels"] = labels.to(device)
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            total_tokens += (batch.get("labels") != -100).sum().item() if "labels" in batch else batch["input_ids"].numel()
            n += 1
    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / n
    return math.exp(min(avg_loss, 100.0))
