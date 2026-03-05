"""Resource utilization tracking (GPU/CPU/memory)."""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ResourceTracker:
    """Track GPU and CPU utilization during training."""

    def __init__(self, use_gpu: bool = True) -> None:
        self.use_gpu = use_gpu
        self._readings: list[dict] = []
        self._gpu_available = False
        try:
            import torch
            self._gpu_available = torch.cuda.is_available()
        except ImportError:
            pass

    def sample(self) -> dict[str, Any]:
        """Take a single sample of resource usage."""
        sample = {"timestamp": time.time(), "cpu_percent": 0.0, "gpu_memory_allocated": 0.0, "gpu_utilization": 0.0}
        try:
            import psutil
            sample["cpu_percent"] = psutil.cpu_percent()
            sample["memory_gb"] = psutil.virtual_memory().used / (1024**3)
        except ImportError:
            pass
        if self._gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    sample["gpu_memory_allocated"] = torch.cuda.memory_allocated() / (1024**3)
                    sample["gpu_memory_reserved"] = torch.cuda.memory_reserved() / (1024**3)
            except Exception:
                pass
        self._readings.append(sample)
        return sample

    def summary(self) -> dict[str, Any]:
        if not self._readings:
            return {}
        gpu_mem = [r["gpu_memory_allocated"] for r in self._readings if "gpu_memory_allocated" in r and r["gpu_memory_allocated"]]
        return {
            "samples": len(self._readings),
            "max_gpu_memory_gb": max(gpu_mem) if gpu_mem else 0.0,
            "avg_cpu_percent": sum(r["cpu_percent"] for r in self._readings) / len(self._readings),
        }
