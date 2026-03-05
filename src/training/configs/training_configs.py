"""Training hyperparameters and run configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Training run configuration."""

    output_dir: str = "./outputs"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: Optional[int] = 100
    save_total_limit: Optional[int] = 3
    deepspeed_config: Optional[str] = None
    dataloader_num_workers: int = 0
    report_to: str = "wandb"  # wandb | mlflow | none

    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
