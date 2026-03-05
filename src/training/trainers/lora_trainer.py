"""LoRA (Low-Rank Adaptation) trainer."""

from typing import Any, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from .base_trainer import BaseTrainer
from ..configs.model_configs import ModelConfig
from ..configs.training_configs import TrainingConfig
from ..configs.peft_configs import LoRAConfig


class LoRATrainer(BaseTrainer):
    """Trainer with LoRA adapters for parameter-efficient fine-tuning."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        lora_config: LoRAConfig,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        callbacks: Optional[list] = None,
    ) -> None:
        self.lora_config = lora_config
        super().__init__(
            model_config=model_config,
            training_config=training_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

    def _load_model_and_tokenizer(self) -> None:
        """Load model, apply LoRA, then set tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model,
            trust_remote_code=self.model_config.trust_remote_code,
            use_fast=self.model_config.use_fast_tokenizer,
        )
        import torch
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            torch_dtype=torch.float16 if self.training_config.fp16 else torch.bfloat16 if self.training_config.bf16 else torch.float32,
            device_map=self.model_config.device_map,
            trust_remote_code=self.model_config.trust_remote_code,
            low_cpu_mem_usage=self.model_config.low_cpu_mem_usage,
        )
        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules,
            bias=self.lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=self.lora_config.inference_mode,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
