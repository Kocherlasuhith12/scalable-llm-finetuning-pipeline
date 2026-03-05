"""Benchmark suite for automated evaluation."""

import logging
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """Run multiple evaluators and aggregate results."""

    def __init__(
        self,
        evaluators: Optional[list[tuple[str, Callable]]] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.evaluators = evaluators or []
        self.output_dir = Path(output_dir) if output_dir else None

    def add_evaluator(self, name: str, fn: Callable) -> None:
        self.evaluators.append((name, fn))

    def run(
        self,
        model: Any,
        tokenizer: Any,
        eval_dataset: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Run all evaluators and return name -> metric dict."""
        results = {}
        for name, fn in self.evaluators:
            try:
                out = fn(model=model, tokenizer=tokenizer, eval_dataset=eval_dataset, **kwargs)
                if isinstance(out, dict):
                    results.update({f"{name}_{k}": v for k, v in out.items()})
                else:
                    results[name] = float(out)
            except Exception as e:
                logger.warning("Evaluator %s failed: %s", name, e)
                results[name] = float("nan")
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            import json
            (self.output_dir / "benchmark_results.json").write_text(json.dumps(results, indent=2))
        return results
