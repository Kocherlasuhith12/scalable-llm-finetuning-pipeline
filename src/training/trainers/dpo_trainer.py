"""DPO (Direct Preference Optimization) trainer for alignment from preferences."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

from ..configs.model_configs import ModelConfig
from ..configs.training_configs import TrainingConfig
from ..configs.peft_configs import LoRAConfig
from ..configs.dpo_configs import DPOConfig

logger = logging.getLogger(__name__)

try:
    from trl import DPOTrainer, DPOConfig as TRLDPOConfig
    _TRL_AVAILABLE = True
except ImportError:
    _TRL_AVAILABLE = False


def _to_trl_dpo_config(
    training_config: TrainingConfig,
    dpo_config: DPOConfig,
) -> "TRLDPOConfig":
    """Build TRL DPOConfig from our configs."""
    from trl import DPOConfig as TRLDPOConfig
    return TRLDPOConfig(
        beta=dpo_config.beta,
        label_smoothing=dpo_config.label_smoothing,
        loss_type=dpo_config.loss_type,
        max_length=dpo_config.max_length,
        max_prompt_length=dpo_config.max_prompt_length,
        max_target_length=dpo_config.max_target_length,
        learning_rate=training_config.learning_rate,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        warmup_ratio=training_config.warmup_ratio,
        weight_decay=training_config.weight_decay,
        max_grad_norm=training_config.max_grad_norm,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        output_dir=training_config.output_dir,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        gradient_checkpointing=training_config.gradient_checkpointing,
        report_to=training_config.report_to or "none",
        remove_unused_columns=False,
    )


class DPOTrainerWrapper:
    """Wrapper around TRL DPOTrainer for our config and dataset format."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        dpo_config: DPOConfig,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        lora_config: Optional[LoRAConfig] = None,
        **kwargs: Any,
    ) -> None:
        if not _TRL_AVAILABLE:
            raise RuntimeError("DPO requires trl. Install with: pip install trl")
        self.model_config = model_config
        self.training_config = training_config
        self.dpo_config = dpo_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.lora_config = lora_config
        self._trainer: Any = None

    def _prepare_dataset(self, dataset: Any):
        """Convert our PreferenceDataset to HuggingFace Dataset for TRL."""
        try:
            from datasets import Dataset
        except ImportError:
            raise RuntimeError("datasets package required for DPO. Install with: pip install datasets")
        if hasattr(dataset, "to_trl_format"):
            rows = dataset.to_trl_format()
        elif hasattr(dataset, "_data"):
            rows = [
                {
                    "prompt": d.get("prompt", d.get("instruction", "")),
                    "chosen": d.get("chosen", d.get("chosen_response", "")),
                    "rejected": d.get("rejected", d.get("rejected_response", "")),
                }
                for d in dataset._data
            ]
        else:
            rows = list(dataset)
        return Dataset.from_list(rows)

    def train(self) -> None:
        """Run DPO training via TRL DPOTrainer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
        import torch

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model,
            trust_remote_code=self.model_config.trust_remote_code,
            use_fast=self.model_config.use_fast_tokenizer,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            torch_dtype=torch.float32,  # safest on CPU/Mac
            device_map=None,
            trust_remote_code=self.model_config.trust_remote_code,
            low_cpu_mem_usage=False,
        )

        if self.training_config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        ref_model = None
        if self.dpo_config.ref_model is not None:
            ref_model = AutoModelForCausalLM.from_pretrained(
                self.dpo_config.ref_model,
                torch_dtype=model.dtype,
                device_map="auto",
                trust_remote_code=self.model_config.trust_remote_code,
            )

        if self.lora_config is not None:
            peft_config = LoraConfig(
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                target_modules=self.lora_config.target_modules,
                bias=self.lora_config.bias,
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        train_ds = self._prepare_dataset(self.train_dataset)
        eval_ds = self._prepare_dataset(self.eval_dataset) if self.eval_dataset else None

        trl_config = _to_trl_dpo_config(self.training_config, self.dpo_config)

        trainer_kw: dict = dict(
            model=model,
            args=trl_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            max_length=self.dpo_config.max_length,
            max_prompt_length=self.dpo_config.max_prompt_length,
            max_target_length=self.dpo_config.max_target_length,
            beta=self.dpo_config.beta,
            loss_type=self.dpo_config.loss_type,
        )
        if ref_model is not None:
            trainer_kw["ref_model"] = ref_model
        trainer_kw["tokenizer"] = tokenizer
        self._trainer = DPOTrainer(**trainer_kw)
        self._trainer.train()
        self._trainer.save_model(Path(self.training_config.output_dir) / "final")
        tokenizer.save_pretrained(Path(self.training_config.output_dir) / "final")
        logger.info("DPO training complete. Model saved to %s/final", self.training_config.output_dir)
