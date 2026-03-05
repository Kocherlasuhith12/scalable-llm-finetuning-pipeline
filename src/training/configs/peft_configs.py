"""PEFT (LoRA/QLoRA) configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False


@dataclass
class QLoRAConfig(LoRAConfig):
    """QLoRA (quantized LoRA) configuration."""

    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
