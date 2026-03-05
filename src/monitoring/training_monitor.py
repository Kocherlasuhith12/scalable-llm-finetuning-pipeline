"""Training progress and metric monitoring."""

import logging
import time
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Track training metrics (loss, throughput, ETA)."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.losses: deque = deque(maxlen=window_size)
        self.start_time: Optional[float] = None
        self.step_times: deque = deque(maxlen=window_size)
        self._last_step = 0.0

    def update(self, step: int, loss: float, **kwargs: Any) -> None:
        now = time.perf_counter()
        if self.start_time is None:
            self.start_time = now
        self.losses.append(loss)
        if self._last_step > 0:
            self.step_times.append(now - self._last_step)
        self._last_step = now

    def current_loss(self) -> float:
        return sum(self.losses) / len(self.losses) if self.losses else 0.0

    def steps_per_second(self) -> float:
        if not self.step_times:
            return 0.0
        return 1.0 / (sum(self.step_times) / len(self.step_times))

    def summary(self) -> dict[str, Any]:
        return {
            "current_loss": self.current_loss(),
            "steps_per_second": self.steps_per_second(),
            "total_steps": len(self.losses),
        }
