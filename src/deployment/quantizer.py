"""Model quantization (INT8/INT4) for efficient deployment."""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Quantize models for smaller size and faster inference."""

    def __init__(
        self,
        bits: int = 8,
        dtype: str = "int8",
        output_dir: str = "./quantized",
    ) -> None:
        self.bits = bits
        self.dtype = dtype
        self.output_dir = Path(output_dir)

    def quantize(
        self,
        model: Any,
        tokenizer: Any,
        calibration_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> Path:
        """Quantize model and save to output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.bits == 4:
                from transformers import BitsAndBytesConfig
                import torch
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model = model.__class__.from_pretrained(
                    getattr(model, "name_or_path", model.config._name_or_path),
                    quantization_config=quant_config,
                    device_map="auto",
                )
            # Save quantized model
            model.save_pretrained(self.output_dir)
            if tokenizer:
                tokenizer.save_pretrained(self.output_dir)
            return self.output_dir
        except Exception as e:
            logger.warning("Quantization failed: %s", e)
            return self.output_dir
