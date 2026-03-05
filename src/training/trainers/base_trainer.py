"""Base trainer for LLM fine-tuning."""

import logging
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from ..configs.model_configs import ModelConfig
from ..configs.training_configs import TrainingConfig
from ..callbacks.checkpoint_callback import CheckpointCallback
from ..callbacks.early_stopping import EarlyStoppingCallback
from ..callbacks.metrics_logger import MetricsLoggerCallback
from ..optimizers.custom_optimizers import get_optimizer, get_scheduler

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base training loop with checkpointing, logging, and early stopping."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        callbacks: Optional[list] = None,
    ) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.callbacks = callbacks or []
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self._global_step = 0

    def _load_model_and_tokenizer(self) -> None:
        """Load base model and tokenizer from config."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model,
            trust_remote_code=self.model_config.trust_remote_code,
            use_fast=self.model_config.use_fast_tokenizer,
        )
        dtype = getattr(torch, self.model_config.torch_dtype, None) if isinstance(self.model_config.torch_dtype, str) and self.model_config.torch_dtype != "auto" else self.model_config.torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            torch_dtype=dtype,
            device_map=self.model_config.device_map,
            trust_remote_code=self.model_config.trust_remote_code,
            low_cpu_mem_usage=self.model_config.low_cpu_mem_usage,
        )
        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def _get_dataloader(self, dataset: Any, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=shuffle,
            num_workers=self.training_config.dataloader_num_workers,
            pin_memory=True,
        )

    def _run_callbacks(self, hook: str, **kwargs: Any) -> None:
        for cb in self.callbacks:
            method = getattr(cb, hook, None)
            if method and callable(method):
                method(**kwargs)

    def train(self) -> None:
        """Run full training loop."""
        self._load_model_and_tokenizer()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer failed to load")
        if hasattr(self.train_dataset, "set_tokenizer"):
            self.train_dataset.set_tokenizer(self.tokenizer)
        train_loader = self._get_dataloader(self.train_dataset)
        eval_loader = self._get_dataloader(self.eval_dataset, shuffle=False) if self.eval_dataset else None
        num_steps = len(train_loader) * self.training_config.num_epochs
        optimizer = get_optimizer(
            self.model,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        scheduler = get_scheduler(
            optimizer,
            num_training_steps=num_steps,
            warmup_ratio=self.training_config.warmup_ratio,
        )
        self._run_callbacks("on_train_begin", config=vars(self.training_config))
        self.model.train()
        for epoch in range(self.training_config.num_epochs):
            for batch in train_loader:
                inputs = {k: v.to(self.model.device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
                labels = batch.get("labels")
                if labels is not None:
                    inputs["labels"] = labels.to(self.model.device)
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()
                if self._global_step % self.training_config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                if self._global_step > 0 and self._global_step % self.training_config.logging_steps == 0:
                    self._run_callbacks("on_log", logs={"loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=self._global_step)
                for cb in self.callbacks:
                    if isinstance(cb, CheckpointCallback):
                        cb.on_step_end(self._global_step, self.model, self.tokenizer)
                self._global_step += 1
            if eval_loader and self.training_config.eval_steps:
                eval_loss = self._evaluate(eval_loader)
                self._run_callbacks("on_log", logs={"eval_loss": eval_loss}, step=self._global_step)
                self._run_callbacks("on_eval_end", metrics={"eval_loss": eval_loss})
                for cb in self.callbacks:
                    if isinstance(cb, EarlyStoppingCallback) and cb.should_stop:
                        logger.info("Early stopping")
                        self._run_callbacks("on_train_end")
                        self._save_final()
                        return
        self._run_callbacks("on_train_end")
        self._save_final()

    def _evaluate(self, eval_loader: DataLoader) -> float:
        if self.model is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in eval_loader:
                inputs = {k: v.to(self.model.device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
                labels = batch.get("labels")
                if labels is not None:
                    inputs["labels"] = labels.to(self.model.device)
                outputs = self.model(**inputs)
                total_loss += outputs.loss.item()
                n += 1
        self.model.train()
        return total_loss / n if n else 0.0

    def _save_final(self) -> None:
        out = Path(self.training_config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.model:
            self.model.save_pretrained(out)
        if self.tokenizer:
            self.tokenizer.save_pretrained(out)
        logger.info("Saved final model to %s", out)
