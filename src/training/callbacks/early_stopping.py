"""Early stopping callback for training."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EarlyStoppingCallback:
    """Stop training when metric stops improving."""

    def __init__(
        self,
        metric: str = "eval_loss",
        patience: int = 3,
        mode: str = "min",  # min for loss, max for accuracy
        min_delta: float = 0.0,
    ) -> None:
        self.metric = metric
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self._counter = 0
        self._best: Optional[float] = None
        self._should_stop = False

    def _is_better(self, current: float, best: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "min":
            return current < (best - self.min_delta)
        return current > (best + self.min_delta)

    def on_eval_end(self, metrics: dict[str, float], **kwargs: Any) -> bool:
        """Called after evaluation. Returns True if should stop."""
        value = metrics.get(self.metric)
        if value is None:
            return False
        if self._is_better(value, self._best or (float("inf") if self.mode == "min" else float("-inf"))):
            self._best = value
            self._counter = 0
            return False
        self._counter += 1
        if self._counter >= self.patience:
            logger.info("Early stopping triggered (metric=%s, patience=%d)", self.metric, self.patience)
            self._should_stop = True
            return True
        return False

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    def reset(self) -> None:
        self._counter = 0
        self._best = None
        self._should_stop = False
