"""DPO (Direct Preference Optimization) and reward model configuration."""

from dataclasses import dataclass, field
from typing import Optional

from .peft_configs import LoRAConfig


@dataclass
class DPOConfig:
    """Configuration for DPO training."""

    beta: float = 0.1  # KL penalty coefficient (higher = stronger reference constraint)
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # sigmoid | hinge | ipo | kto
    max_length: int = 1024
    max_prompt_length: int = 512
    max_target_length: Optional[int] = None  # defaults to max_length - max_prompt_length
    ref_model: Optional[str] = None  # path or same as base; None = use non-trainable copy
    ref_model_adapters: Optional[str] = None
    precompute_ref_log_probs: bool = False
    dataset_num_proc: Optional[int] = None


@dataclass
class RewardModelConfig:
    """Configuration for reward model training (optional stage before/alongside DPO)."""

    num_train_epochs: int = 1
    max_length: int = 1024
    # Loss: typically pairwise ranking (chosen vs rejected)
    loss_type: str = "rank"  # rank | token_level
    pooling: str = "last"  # last | mean | max
    num_negatives: int = 1  # rejected per prompt
