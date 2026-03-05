"""Instruction-tuning dataset with prompt templates."""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from transformers import PreTrainedTokenizer

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class InstructionDataset(BaseDataset):
    """Dataset for instruction/response pairs (SFT format)."""

    DEFAULT_TEMPLATE = (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"
    )

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 2048,
        instruction_key: str = "instruction",
        input_key: str = "input",
        response_key: str = "output",
        template: Optional[str] = None,
        max_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_key = instruction_key
        self.input_key = input_key
        self.response_key = response_key
        self.template = template or self.DEFAULT_TEMPLATE
        self._tokenized: list[dict[str, Any]] = []
        super().__init__(data_path=data_path, max_samples=max_samples, **kwargs)
        if tokenizer and self._data:
            self._tokenize_all()

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
            items = raw if isinstance(raw, list) else raw.get("data", raw.get("instances", []))
            for i, item in enumerate(items):
                if self.max_samples and i >= self.max_samples:
                    break
                self._data.append(item if isinstance(item, dict) else {"instruction": str(item)})
        else:
            logger.warning("Unsupported format: %s", path.suffix)

    def _format_prompt(self, item: dict) -> str:
        instruction = item.get(self.instruction_key, "")
        input_text = item.get(self.input_key, "")
        response = item.get(self.response_key, "")
        return self.template.format(
            instruction=instruction,
            input=input_text,
            response=response,
        )

    def _tokenize_all(self) -> None:
        if not self.tokenizer:
            return
        for item in self._data:
            text = self._format_prompt(item)
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )
            enc["labels"] = enc["input_ids"].copy()
            self._tokenized.append(enc)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._tokenized:
            return self._tokenized[index]
        item = self._data[index]
        text = self._format_prompt(item)
        if self.tokenizer:
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )
            enc["labels"] = enc["input_ids"].copy()
            return enc
        return {"text": text, "raw": item}

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        """Set tokenizer and (re)tokenize."""
        self.tokenizer = tokenizer
        self._tokenized = []
        if self._data:
            self._tokenize_all()
