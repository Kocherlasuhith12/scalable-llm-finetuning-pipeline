#!/usr/bin/env python3
"""Run DPO (Direct Preference Optimization) training from config and preference data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="DPO alignment training")
    parser.add_argument("--config", type=str, default="configs/dpo_config.yaml")
    parser.add_argument("--data-path", type=str, default=None, help="Path to preference JSONL (prompt, chosen, rejected)")
    parser.add_argument("--output-dir", type=str, default="outputs/dpo")
    parser.add_argument("--train-reward-model", action="store_true", help="Optionally train a reward model first")
    args = parser.parse_args()

    from workflows.dags.dpo_pipeline import run_dpo_pipeline

    out = run_dpo_pipeline(
        config_path=args.config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        train_reward_model=args.train_reward_model,
    )
    print(f"DPO training complete. Model saved to: {out}")


if __name__ == "__main__":
    main()
