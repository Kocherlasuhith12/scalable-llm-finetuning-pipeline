"""Visualization utilities for evaluation and training."""

from pathlib import Path
from typing import Any, Optional

try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: Optional[Path] = None,
    title: str = "Training",
) -> None:
    """Plot loss/accuracy curves from history."""
    if not _MATPLOTLIB:
        return
    fig, ax = plt.subplots()
    for name, values in history.items():
        ax.plot(values, label=name)
    ax.set_xlabel("Step")
    ax.legend()
    ax.set_title(title)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    plt.close(fig)


def plot_metrics_comparison(
    runs: dict[str, dict[str, float]],
    metric_names: Optional[list[str]] = None,
    output_path: Optional[Path] = None,
) -> None:
    """Bar chart comparing metrics across runs."""
    if not _MATPLOTLIB or not runs:
        return
    metric_names = metric_names or list(next(iter(runs.values())).keys())
    run_names = list(runs.keys())
    x = range(len(run_names))
    width = 0.8 / len(metric_names)
    fig, ax = plt.subplots()
    for i, m in enumerate(metric_names):
        vals = [runs[r].get(m, 0) for r in run_names]
        ax.bar([xi + i * width for xi in x], vals, width, label=m)
    ax.set_xticks([xi + width * (len(metric_names) - 1) / 2 for xi in x])
    ax.set_xticklabels(run_names)
    ax.legend()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    plt.close(fig)
