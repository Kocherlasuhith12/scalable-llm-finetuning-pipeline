"""Cost estimation for training runs."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Example hourly rates (USD) - override via config
DEFAULT_RATES = {
    "local": 0.0,
    "aws_p4d": 32.0,
    "aws_p3": 3.06,
    "gcp_a100": 2.93,
    "azure_a100": 3.67,
}


class CostCalculator:
    """Estimate cost of training based on compute and duration."""

    def __init__(
        self,
        instance_type: str = "local",
        hourly_rates: Optional[dict[str, float]] = None,
    ) -> None:
        self.instance_type = instance_type
        self.rates = hourly_rates or DEFAULT_RATES
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self) -> None:
        import time
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        import time
        self.end_time = time.perf_counter()

    def estimate(self, num_gpus: int = 1) -> dict[str, Any]:
        """Return estimated cost for the run."""
        if self.start_time is None or self.end_time is None:
            duration_hours = 0.0
        else:
            duration_hours = (self.end_time - self.start_time) / 3600.0
        rate = self.rates.get(self.instance_type, 0.0)
        cost = duration_hours * rate * num_gpus
        return {
            "duration_hours": duration_hours,
            "instance_type": self.instance_type,
            "num_gpus": num_gpus,
            "hourly_rate_usd": rate,
            "estimated_cost_usd": round(cost, 2),
        }
