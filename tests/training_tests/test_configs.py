"""Tests for training configs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.configs.training_configs import TrainingConfig
from src.training.configs.peft_configs import LoRAConfig

def test_training_config_effective_batch():
    cfg = TrainingConfig(batch_size=4, gradient_accumulation_steps=4)
    assert cfg.effective_batch_size() == 16

def test_lora_config_defaults():
    cfg = LoRAConfig()
    assert cfg.r == 16
    assert "q_proj" in cfg.target_modules
