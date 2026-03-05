"""Training pipeline DAG: load data, train, checkpoint, log."""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def run_training_pipeline(
    config_path: str = "configs/lora_config.yaml",
    data_path: Optional[str] = None,
    output_dir: str = "outputs",
    **kwargs: Any,
) -> str:
    """Run training from config; return path to final model."""
    from src.utils.config_parser import load_config
    from src.training.configs.model_configs import ModelConfig
    from src.training.configs.training_configs import TrainingConfig
    from src.training.configs.peft_configs import LoRAConfig, QLoRAConfig
    from src.training.trainers.lora_trainer import LoRATrainer
    from src.training.trainers.qlora_trainer import QLoRATrainer
    from src.training.callbacks.checkpoint_callback import CheckpointCallback
    from src.training.callbacks.metrics_logger import MetricsLoggerCallback
    from src.data.datasets.instruction_dataset import InstructionDataset

    config = load_config(config_path)
    train_cfg = config.get("training", {})
    peft_cfg = config.get("peft", {})
    model_cfg = train_cfg.get("model", {})

    data_path = data_path or "data/processed/train.jsonl"
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    model_config = ModelConfig(
        base_model=model_cfg.get("base_model", "meta-llama/Llama-2-7b-hf"),
        use_peft=model_cfg.get("use_peft", True),
        peft_type=model_cfg.get("peft_type", "lora"),
    )
    training_config = TrainingConfig(
        output_dir=output_dir,
        num_epochs=train_cfg.get("num_epochs", 3),
        batch_size=train_cfg.get("batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        save_steps=train_cfg.get("save_steps", 100),
        logging_steps=train_cfg.get("logging_steps", 10),
        report_to=config.get("project", {}).get("experiment_tracking", "wandb"),
    )
    training_config.output_dir = output_dir

    dataset = InstructionDataset(data_path=data_path, max_length=train_cfg.get("max_sequence_length", 2048))
    callbacks = [
        CheckpointCallback(output_dir=output_dir, save_steps=training_config.save_steps, save_total_limit=training_config.save_total_limit),
        MetricsLoggerCallback(backend=training_config.report_to, project=config.get("project", {}).get("name", "llm-finetuning")),
    ]

    peft_type = model_cfg.get("peft_type", "lora")
    if peft_type == "qlora":
        qlora = QLoRAConfig(
            load_in_4bit=peft_cfg.get("qlora", {}).get("load_in_4bit", True),
            r=peft_cfg.get("lora", {}).get("r", 16),
            lora_alpha=peft_cfg.get("lora", {}).get("lora_alpha", 32),
            target_modules=peft_cfg.get("lora", {}).get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        )
        trainer = QLoRATrainer(model_config, training_config, qlora, dataset, callbacks=callbacks)
    else:
        lora = LoRAConfig(
            r=peft_cfg.get("lora", {}).get("r", 16),
            lora_alpha=peft_cfg.get("lora", {}).get("lora_alpha", 32),
            target_modules=peft_cfg.get("lora", {}).get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        )
        trainer = LoRATrainer(model_config, training_config, lora, dataset, callbacks=callbacks)

    trainer.train()
    return str(Path(output_dir).resolve())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_config.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run_training_pipeline(config_path=args.config, data_path=args.data_path, output_dir=args.output_dir)
