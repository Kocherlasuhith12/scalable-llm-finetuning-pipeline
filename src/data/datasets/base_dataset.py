"""Base dataset abstraction for LLM training."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base dataset for tokenized LLM data."""

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path) if data_path else None
        self.max_samples = max_samples
        self._data: list[dict[str, Any]] = []
        if self.data_path and self.data_path.exists():
            self._load()

    @abstractmethod
    def _load(self) -> None:
        """Load data from path or source."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return single sample (e.g. input_ids, attention_mask, labels)."""
        pass

    def __len__(self) -> int:
        return len(self._data)

    def iter_raw(self) -> Iterator[dict[str, Any]]:
        """Iterate over raw records before tokenization."""
        for i in range(len(self._data)):
            yield self._data[i]
