"""DPO (Direct Preference Optimization) pipeline: train on preference data."""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def run_dpo_pipeline(
    config_path: str = "configs/dpo_config.yaml",
    data_path: Optional[str] = None,
    output_dir: str = "outputs/dpo",
    train_reward_model: bool = False,
    **kwargs: Any,
) -> str:
    """Run DPO training on preference data. Optionally train reward model first."""
    from src.utils.config_parser import load_config
    from src.training.configs.model_configs import ModelConfig
    from src.training.configs.training_configs import TrainingConfig
    from src.training.configs.peft_configs import LoRAConfig
    from src.training.configs.dpo_configs import DPOConfig, RewardModelConfig
    from src.training.trainers.dpo_trainer import DPOTrainerWrapper
    from src.training.trainers.reward_model_trainer import RewardModelTrainer
    from src.data.datasets.preference_dataset import PreferenceDataset

    config = load_config(config_path)
    train_cfg = config.get("training", {})
    dpo_cfg = config.get("dpo", {})
    peft_cfg = config.get("peft", {}).get("lora", {})

    data_path = data_path or config.get("data", {}).get("preference_data_path", "data/processed/preferences.jsonl")
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Preference data not found: {data_path}. "
            "Create a JSONL with lines: {\"prompt\": \"...\", \"chosen\": \"...\", \"rejected\": \"...\"}"
        )

    model_config = ModelConfig(
        base_model=train_cfg.get("model", {}).get("base_model", "meta-llama/Llama-2-7b-hf"),
        use_peft=True,
    )
    training_config = TrainingConfig(
        output_dir=output_dir,
        num_epochs=train_cfg.get("num_epochs", 1),
        batch_size=train_cfg.get("batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 5e-7),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        save_steps=train_cfg.get("save_steps", 100),
        logging_steps=train_cfg.get("logging_steps", 10),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        report_to=config.get("project", {}).get("experiment_tracking", "wandb"),
    )
    training_config.output_dir = output_dir

    dpo_config = DPOConfig(
        beta=dpo_cfg.get("beta", 0.1),
        loss_type=dpo_cfg.get("loss_type", "sigmoid"),
        max_length=dpo_cfg.get("max_length", 1024),
        max_prompt_length=dpo_cfg.get("max_prompt_length", 512),
        ref_model=dpo_cfg.get("ref_model"),
    )
    lora_config = LoRAConfig(
        r=peft_cfg.get("r", 16),
        lora_alpha=peft_cfg.get("lora_alpha", 32),
        lora_dropout=peft_cfg.get("lora_dropout", 0.05),
        target_modules=peft_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
    )

    dataset = PreferenceDataset(data_path=data_path, max_length=dpo_config.max_length)

    if train_reward_model:
        rm_config = RewardModelConfig(max_length=dpo_config.max_length)
        rm_trainer = RewardModelTrainer(
            model_config=model_config,
            training_config=TrainingConfig(
                output_dir=str(Path(output_dir) / "reward_model"),
                batch_size=training_config.batch_size,
                learning_rate=1e-5,
                num_epochs=1,
                logging_steps=10,
            ),
            reward_config=rm_config,
            train_dataset=dataset,
        )
        rm_trainer.train()

    trainer = DPOTrainerWrapper(
        model_config=model_config,
        training_config=training_config,
        dpo_config=dpo_config,
        train_dataset=dataset,
        lora_config=lora_config,
    )
    trainer.train()
    return str(Path(output_dir).resolve())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo_config.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default="outputs/dpo")
    parser.add_argument("--train-reward-model", action="store_true", help="Train reward model first (optional)")
    args = parser.parse_args()
    run_dpo_pipeline(
        config_path=args.config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        train_reward_model=args.train_reward_model,
    )
