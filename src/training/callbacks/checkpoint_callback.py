"""Checkpoint saving callback for training."""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CheckpointCallback:
    """Callback to manage checkpoints (save/rotate)."""

    def __init__(
        self,
        output_dir: str = "./checkpoints",
        save_total_limit: Optional[int] = 3,
        save_steps: int = 100,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.save_total_limit = save_total_limit
        self.save_steps = save_steps
        self._saved_dirs: list[Path] = []

    def on_step_end(
        self,
        step: int,
        model: Any,
        tokenizer: Any,
        **kwargs: Any,
    ) -> None:
        """Called at end of step; save if at save_steps boundary."""
        if step > 0 and step % self.save_steps == 0:
            self.save_checkpoint(step, model, tokenizer, **kwargs)

    def save_checkpoint(
        self,
        step: int,
        model: Any,
        tokenizer: Any,
        **kwargs: Any,
    ) -> Path:
        """Save model/tokenizer to output_dir/checkpoint-{step}."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(ckpt_dir)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(ckpt_dir)
        self._saved_dirs.append(ckpt_dir)
        if self.save_total_limit and len(self._saved_dirs) > self.save_total_limit:
            to_remove = self._saved_dirs.pop(0)
            import shutil
            if to_remove.exists():
                shutil.rmtree(to_remove)
        logger.info("Saved checkpoint to %s", ckpt_dir)
        return ckpt_dir
