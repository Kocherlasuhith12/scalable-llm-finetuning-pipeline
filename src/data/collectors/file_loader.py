"""File-based data loader for local and cloud storage."""

import json
import logging
from pathlib import Path
from typing import Any, Iterator, Optional, Union

logger = logging.getLogger(__name__)


class FileLoader:
    """Load training data from local files or cloud storage (S3/GCS/Azure)."""

    SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv", ".txt", ".parquet"}

    def __init__(
        self,
        storage_type: str = "local",
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> None:
        self.storage_type = storage_type
        self.bucket = bucket
        self.prefix = prefix or ""

    def _load_local(self, path: Path) -> Iterator[dict[str, Any]]:
        """Load from local filesystem."""
        if path.is_file():
            yield from self._read_file(path)
        elif path.is_dir():
            for ext in self.SUPPORTED_EXTENSIONS:
                for f in path.rglob(f"*{ext}"):
                    yield from self._read_file(f)
        else:
            logger.warning("Path does not exist: %s", path)

    def _read_file(self, path: Path) -> Iterator[dict[str, Any]]:
        """Read a single file and yield document dicts."""
        suffix = path.suffix.lower()
        try:
            if suffix == ".jsonl":
                with open(path, encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            obj.setdefault("id", f"{path.stem}_{i}")
                            yield obj
                        else:
                            yield {"id": f"{path.stem}_{i}", "text": str(obj)}
            elif suffix == ".json":
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else data.get("data", [data])
                for i, item in enumerate(items):
                    if isinstance(item, dict):
                        item.setdefault("id", f"{path.stem}_{i}")
                        yield item
                    else:
                        yield {"id": f"{path.stem}_{i}", "text": str(item)}
            elif suffix == ".txt":
                text = path.read_text(encoding="utf-8")
                yield {"id": path.stem, "text": text, "source": str(path)}
            elif suffix == ".csv":
                import csv
                with open(path, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        row["id"] = row.get("id", f"{path.stem}_{i}")
                        yield row
            elif suffix == ".parquet":
                import pandas as pd
                df = pd.read_parquet(path)
                for i, row in df.iterrows():
                    yield {"id": f"{path.stem}_{i}", **row.to_dict()}
        except Exception as e:
            logger.warning("Error reading %s: %s", path, e)

    def load(
        self,
        path: Union[str, Path],
        pattern: Optional[str] = None,
    ) -> Iterator[dict[str, Any]]:
        """Load documents from path (local or cloud)."""
        path = Path(path)
        if self.storage_type == "local":
            yield from self._load_local(path)
        else:
            yield from self._load_cloud(path, pattern)

    def _load_cloud(
        self,
        path: Union[str, Path],
        pattern: Optional[str],
    ) -> Iterator[dict[str, Any]]:
        """Load from S3/GCS/Azure; delegates to local cache if synced."""
        # Placeholder: in production, use boto3/gcs_client/azure blob
        logger.info("Cloud load requested for %s (bucket=%s)", path, self.bucket)
        local_cache = Path("data/cache") / self.prefix
        if local_cache.exists():
            yield from self._load_local(local_cache)
        else:
            logger.warning("Cloud storage not implemented; use local path or sync first.")
