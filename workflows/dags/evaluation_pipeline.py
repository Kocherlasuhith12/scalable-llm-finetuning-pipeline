"""Evaluation pipeline DAG: load checkpoint, run benchmarks, log results."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def run_evaluation_pipeline(
    checkpoint_dir: str,
    config_path: str = "configs/base_config.yaml",
    output_dir: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Evaluate a trained checkpoint; return metrics dict."""
    from src.utils.config_parser import load_config
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader
    from src.evaluation.evaluators.perplexity import compute_perplexity
    from src.evaluation.evaluators.benchmark_suite import BenchmarkSuite
    from src.data.datasets.instruction_dataset import InstructionDataset

    config = load_config(config_path)
    ckpt = Path(checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(checkpoint_dir)
    output_dir = output_dir or str(ckpt / "eval")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(ckpt, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    eval_cfg = config.get("evaluation", {})
    eval_data = eval_cfg.get("eval_dataset") or "data/processed/val.jsonl"
    if Path(eval_data).exists():
        dataset = InstructionDataset(eval_data, tokenizer=tokenizer, max_length=eval_cfg.get("max_length", 2048))
        loader = DataLoader(dataset, batch_size=eval_cfg.get("eval_batch_size", 8))
        ppl = compute_perplexity(model, loader)
    else:
        ppl = float("nan")

    results = {"perplexity": ppl}
    suite = BenchmarkSuite(output_dir=Path(output_dir))
    suite.add_evaluator("perplexity", lambda **kw: ppl)
    suite.run(model=model, tokenizer=tokenizer)

    out_path = Path(output_dir) / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Evaluation results: %s", results)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run_evaluation_pipeline(checkpoint_dir=args.checkpoint_dir, config_path=args.config, output_dir=args.output_dir)
