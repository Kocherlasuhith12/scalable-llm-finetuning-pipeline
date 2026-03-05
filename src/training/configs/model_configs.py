"""Model configuration for fine-tuning."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for base model loading."""

    base_model: str = "meta-llama/Llama-2-7b-hf"
    use_peft: bool = True
    peft_type: Optional[str] = None  # lora | qlora
    torch_dtype: Optional[str] = "auto"
    device_map: Optional[str] = None
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True
    low_cpu_mem_usage: bool = True
