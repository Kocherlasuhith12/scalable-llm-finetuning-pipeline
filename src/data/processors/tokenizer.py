"""Dataset tokenization for LLM training."""

import logging
from typing import Any, Optional, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class DatasetTokenizer:
    """Tokenize text or instruction datasets for training."""

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_length: int = 2048,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        text_column: str = "text",
        add_special_tokens: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.text_column = text_column
        self.add_special_tokens = add_special_tokens

    def tokenize_text(self, text: str) -> dict[str, Any]:
        """Tokenize a single text string."""
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors or "pt",
            add_special_tokens=self.add_special_tokens,
        )
        enc["labels"] = enc["input_ids"].clone()
        return {k: v.squeeze(0) if self.return_tensors == "pt" and v.dim() == 2 else v for k, v in enc.items()}

    def tokenize_instruction(
        self,
        instruction: str,
        input_text: Optional[str] = None,
        output_text: str = "",
        template: Optional[str] = None,
    ) -> dict[str, Any]:
        """Tokenize instruction/input/output for SFT."""
        if template:
            prompt = template.format(
                instruction=instruction,
                input=input_text or "",
                output=output_text,
            )
        else:
            prompt = f"### Instruction:\n{instruction}\n\n"
            if input_text:
                prompt += f"### Input:\n{input_text}\n\n"
            prompt += f"### Response:\n{output_text}"
        return self.tokenize_text(prompt)

    def tokenize_batch(self, examples: dict[str, list]) -> dict[str, Any]:
        """Tokenize a batch of examples (e.g. from datasets)."""
        texts = examples.get(self.text_column, examples.get("content", []))
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
            add_special_tokens=self.add_special_tokens,
        )
