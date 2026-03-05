"""QLoRA (Quantized LoRA) trainer for memory-efficient fine-tuning."""

from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from .base_trainer import BaseTrainer
from ..configs.model_configs import ModelConfig
from ..configs.training_configs import TrainingConfig
from ..configs.peft_configs import QLoRAConfig


class QLoRATrainer(BaseTrainer):
    """Trainer with 4-bit quantization + LoRA for low-memory training."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        qlora_config: QLoRAConfig,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        callbacks: Optional[list] = None,
    ) -> None:
        self.qlora_config = qlora_config
        super().__init__(
            model_config=model_config,
            training_config=training_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

    def _load_model_and_tokenizer(self) -> None:
        """Load model in 4-bit, prepare for k-bit training, apply LoRA."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model,
            trust_remote_code=self.model_config.trust_remote_code,
            use_fast=self.model_config.use_fast_tokenizer,
        )
        compute_dtype = getattr(torch, self.qlora_config.bnb_4bit_compute_dtype, torch.bfloat16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.qlora_config.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.qlora_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.qlora_config.bnb_4bit_use_double_quant,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            quantization_config=bnb_config,
            device_map=self.model_config.device_map,
            trust_remote_code=self.model_config.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        self.model = prepare_model_for_kbit_training(self.model)
        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        peft_config = LoraConfig(
            r=self.qlora_config.r,
            lora_alpha=self.qlora_config.lora_alpha,
            lora_dropout=self.qlora_config.lora_dropout,
            target_modules=self.qlora_config.target_modules,
            bias=self.qlora_config.bias,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=self.qlora_config.inference_mode,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
