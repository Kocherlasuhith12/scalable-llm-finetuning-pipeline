"""Reward model trainer for preference data (optional stage before PPO or for scoring)."""

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from ..configs.model_configs import ModelConfig
from ..configs.training_configs import TrainingConfig
from ..configs.dpo_configs import RewardModelConfig

logger = logging.getLogger(__name__)


class RewardModelTrainer:
    """Train a reward model on (prompt, chosen, rejected) to predict preference."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        reward_config: RewardModelConfig,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.reward_config = reward_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer = None

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            torch_dtype=torch.bfloat16 if self.training_config.bf16 else torch.float32,
            device_map="auto",
            trust_remote_code=self.model_config.trust_remote_code,
        )
        self.model.config.output_hidden_states = True
        # Reward head: project last hidden state to scalar
        hidden_size = self.model.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ).to(self.model.dtype).to(self.model.device)

    def _tokenize_batch(self, chosen: list[str], rejected: list[str], prompts: list[str]):
        """Tokenize chosen and rejected completions with shared prompt."""
        max_len = self.reward_config.max_length
        chosen_tensors = self.tokenizer(
            [p + c for p, c in zip(prompts, chosen)],
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        rejected_tensors = self.tokenizer(
            [p + r for p, r in zip(prompts, rejected)],
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        return chosen_tensors, rejected_tensors

    def train(self) -> None:
        """Train reward model with pairwise ranking loss."""
        self._load_model()
        if self.model is None:
            raise RuntimeError("Model failed to load")
        from torch.optim import AdamW
        from ..optimizers.custom_optimizers import get_scheduler

        opt = AdamW(
            list(self.model.parameters()) + list(self.reward_head.parameters()),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )
        num_steps = len(train_loader) * self.reward_config.num_train_epochs
        scheduler = get_scheduler(opt, num_steps, self.training_config.warmup_ratio)
        self.model.train()
        self.reward_head.train()
        step = 0
        for epoch in range(self.reward_config.num_train_epochs):
            for batch in train_loader:
                prompts = [b["prompt"] for b in batch]
                chosen = [b["chosen"] for b in batch]
                rejected = [b["rejected"] for b in batch]
                chosen_inp, rejected_inp = self._tokenize_batch(chosen, rejected, prompts)
                chosen_inp = {k: v.to(self.model.device) for k, v in chosen_inp.items()}
                rejected_inp = {k: v.to(self.model.device) for k, v in rejected_inp.items()}
                chosen_out = self.model(**chosen_inp, output_hidden_states=True)
                rejected_out = self.model(**rejected_inp, output_hidden_states=True)
                chosen_h = chosen_out.hidden_states[-1]
                rejected_h = rejected_out.hidden_states[-1]
                if self.reward_config.pooling == "last":
                    chosen_h = chosen_h[:, -1, :]
                    rejected_h = rejected_h[:, -1, :]
                else:
                    chosen_h = chosen_h.mean(dim=1)
                    rejected_h = rejected_h.mean(dim=1)
                r_chosen = self.reward_head(chosen_h).squeeze(-1)
                r_rejected = self.reward_head(rejected_h).squeeze(-1)
                loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.reward_head.parameters()),
                    self.training_config.max_grad_norm,
                )
                opt.step()
                scheduler.step()
                opt.zero_grad()
                step += 1
                if step % self.training_config.logging_steps == 0:
                    logger.info("Reward model step %d loss %.4f", step, loss.item())
        out = Path(self.training_config.output_dir) / "reward_model"
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        torch.save(self.reward_head.state_dict(), out / "reward_head.pt")
        logger.info("Reward model saved to %s", out)
