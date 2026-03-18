#!/usr/bin/env python3
"""Evaluate one or more checkpoints and optionally compare metrics."""

import argparse
import json
import sys
from pathlib import Path

# Allow running from project roots
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_parser import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate training checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to checkpoint or directory of checkpoints")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--compare", action="store_true", help="If checkpoint-dir contains multiple checkpoints, compare them")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint_dir)
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} does not exist", file=sys.stderr)
        sys.exit(1)

    from workflows.dags.evaluation_pipeline import run_evaluation_pipeline

    if ckpt_path.is_dir() and (ckpt_path / "config.json").exists():
        # Single checkpoint
        results = run_evaluation_pipeline(
            checkpoint_dir=str(ckpt_path),
            config_path=args.config,
            output_dir=args.output_dir,
        )
        print(json.dumps(results, indent=2))
        return

    if args.compare and ckpt_path.is_dir():
        subdirs = sorted([d for d in ckpt_path.iterdir() if d.is_dir() and (d / "config.json").exists()])
        all_results = {}
        for d in subdirs:
            results = run_evaluation_pipeline(str(d), config_path=args.config, output_dir=str(d / "eval"))
            all_results[d.name] = results
        print(json.dumps(all_results, indent=2))
        return

    results = run_evaluation_pipeline(
        checkpoint_dir=str(ckpt_path),
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
