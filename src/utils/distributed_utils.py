"""Distributed training utilities (Ray, Horovod, DeepSpeed)."""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_rank() -> int:
    """Get global rank (DeepSpeed, torchrun, or env)."""
    for var in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        v = os.environ.get(var)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 0


def get_world_size() -> int:
    """Get world size (number of processes)."""
    for var in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS"):
        v = os.environ.get(var)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: Optional[str] = None) -> None:
    """Initialize distributed backend (torch)."""
    try:
        import torch.distributed as dist
        if not dist.is_initialized() and backend:
            dist.init_process_group(backend=backend or "nccl")
    except Exception as e:
        logger.debug("Distributed init skipped: %s", e)
