"""Preference dataset for DPO/RLHF: (prompt, chosen, rejected) pairs."""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class PreferenceDataset(BaseDataset):
    """Dataset of (prompt, chosen_response, rejected_response) for preference learning."""

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        # Alternative keys for instruction format
        instruction_key: str = "instruction",
        chosen_response_key: str = "chosen_response",
        rejected_response_key: str = "rejected_response",
        max_length: int = 2048,
        max_prompt_length: Optional[int] = None,
        max_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.instruction_key = instruction_key
        self.chosen_response_key = chosen_response_key
        self.rejected_response_key = rejected_response_key
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length or max_length // 2
        super().__init__(data_path=data_path, max_samples=max_samples, **kwargs)

    def _load(self) -> None:
        path = self.data_path
        if not path or not path.exists():
            return
        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if self.max_samples and i >= self.max_samples:
                        break
                    line = line.strip()
                    if line:
                        self._data.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            items = raw if isinstance(raw, list) else raw.get("data", raw.get("preferences", []))
            for i, item in enumerate(items):
                if self.max_samples and i >= self.max_samples:
                    break
                self._data.append(item if isinstance(item, dict) else {"prompt": "", "chosen": str(item), "rejected": ""})
        else:
            logger.warning("Unsupported format: %s", path.suffix)

    def _get_triple(self, item: dict) -> tuple[str, str, str]:
        """Return (prompt, chosen, rejected) from a record."""
        prompt = item.get(self.prompt_key) or item.get("instruction", "")
        chosen = item.get(self.chosen_key) or item.get(self.chosen_response_key, "")
        rejected = item.get(self.rejected_key) or item.get(self.rejected_response_key, "")
        if not prompt and "instruction" in item:
            prompt = item["instruction"]
        return prompt, chosen, rejected

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._data[index]
        prompt, chosen, rejected = self._get_triple(item)
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "raw": item,
        }

    def to_trl_format(self) -> list[dict[str, str]]:
        """Return list of dicts with keys prompt, chosen, rejected for TRL DPOTrainer."""
        return [
            {"prompt": p, "chosen": c, "rejected": r}
            for p, c, r in (self._get_triple(d) for d in self._data)
        ]
