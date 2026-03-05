"""Metrics logging callback (W&B / MLflow)."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MetricsLoggerCallback:
    """Log training/eval metrics to experiment tracker."""

    def __init__(
        self,
        backend: str = "wandb",
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        self.backend = backend
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self._run: Any = None

    def on_train_begin(self, **kwargs: Any) -> None:
        """Initialize experiment run."""
        if self.backend == "wandb":
            try:
                import wandb
                self._run = wandb.init(project=self.project, name=self.run_name, config=self.config)
            except Exception as e:
                logger.warning("wandb init failed: %s", e)
                self._run = None
        elif self.backend == "mlflow":
            try:
                import mlflow
                mlflow.set_experiment(self.project or "llm-finetuning")
                self._run = mlflow.start_run(run_name=self.run_name)
                if self.config:
                    mlflow.log_params(self.config)
            except Exception as e:
                logger.warning("mlflow init failed: %s", e)
                self._run = None

    def on_log(self, logs: dict[str, float], step: Optional[int] = None, **kwargs: Any) -> None:
        """Log metrics."""
        if self.backend == "wandb" and self._run is not None:
            import wandb
            wandb.log(logs, step=step)
        elif self.backend == "mlflow":
            try:
                import mlflow
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v, step=step or 0)
            except Exception as e:
                logger.debug("mlflow log failed: %s", e)

    def on_train_end(self, **kwargs: Any) -> None:
        """Finish run."""
        if self.backend == "wandb" and self._run is not None:
            import wandb
            wandb.finish()
        elif self.backend == "mlflow":
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass
