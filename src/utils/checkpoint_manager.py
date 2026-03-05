"""Checkpoint rotation and management."""

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage checkpoint directories (save, list, prune)."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: Optional[int] = 3,
        prefix: str = "checkpoint-",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.prefix = prefix

    def list_checkpoints(self) -> list[Path]:
        """Return sorted list of checkpoint dirs (oldest first)."""
        if not self.checkpoint_dir.exists():
            return []
        dirs = [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith(self.prefix)]
        return sorted(dirs, key=lambda d: int(d.name.replace(self.prefix, "") or 0))

    def prune(self) -> None:
        """Remove oldest checkpoints if over max_checkpoints."""
        checkpoints = self.list_checkpoints()
        while self.max_checkpoints and len(checkpoints) > self.max_checkpoints:
            to_remove = checkpoints.pop(0)
            shutil.rmtree(to_remove, ignore_errors=True)
            logger.info("Pruned checkpoint %s", to_remove)
            checkpoints = self.list_checkpoints()

    def latest(self) -> Optional[Path]:
        """Return path to latest checkpoint."""
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None
