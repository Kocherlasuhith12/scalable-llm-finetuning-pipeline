"""Custom optimizers and schedulers for fine-tuning."""

from typing import Any, Optional

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw",
    **kwargs: Any,
) -> Optimizer:
    """Build optimizer, optionally with decay only for non-bias/non-LayerNorm params."""
    if optimizer_type.lower() == "adamw":
        # Decay only 2D params that are not bias/LayerNorm
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        params = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        return AdamW(params, lr=learning_rate, **kwargs)
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    raise ValueError(f"Unknown optimizer: {optimizer_type}")


def get_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.03,
    scheduler_type: str = "cosine",
) -> Any:
    """Build LR scheduler with linear warmup."""
    warmup_steps = int(num_training_steps * warmup_ratio)
    if warmup_steps <= 0:
        if scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer, T_max=num_training_steps)
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_training_steps)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    if scheduler_type == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=num_training_steps - warmup_steps)
    else:
        main = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_training_steps - warmup_steps)
    return SequentialLR(optimizer, [warmup, main], milestones=[warmup_steps])
