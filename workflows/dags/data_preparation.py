"""Data preparation DAG: collect, clean, validate, and export dataset."""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def run_data_preparation( 
    config_path: str = "configs/base_config.yaml",
    output_dir: str = "data/processed",
    **kwargs: Any,
) -> str:
    """Run full data prep pipeline and return path to processed dataset."""
    from src.utils.config_parser import load_config
    from src.data.collectors.file_loader import FileLoader
    from src.data.processors.cleaner import TextCleaner
    from src.data.validators.quality_checker import QualityChecker
    from src.data.validators.deduplicator import Deduplicator

    config = load_config(config_path)
    data_cfg = config.get("data", {})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    loader = FileLoader(
        storage_type=config.get("storage", {}).get("type", "local"),
        bucket=config.get("storage", {}).get("bucket"),
        prefix=config.get("storage", {}).get("prefix"),
    )
    cleaner = TextCleaner(
        min_length=data_cfg.get("preprocessing", {}).get("min_sequence_length", 32),
        max_length=data_cfg.get("preprocessing", {}).get("max_sequence_length", 2048),
    )
    quality = QualityChecker()
    dedupe = Deduplicator(threshold=data_cfg.get("quality", {}).get("dedupe_threshold", 0.9))

    sources = data_cfg.get("sources", [])
    raw_docs = []
    for src in sources:
        if src.get("type") == "file":
            path = src.get("path", "data/raw")
            for doc in loader.load(path):
                cleaned = cleaner.clean_document(doc)
                if cleaned:
                    filtered = quality.filter_document(cleaned, min_score=data_cfg.get("quality", {}).get("min_quality_score", 0.5))
                    if filtered:
                        raw_docs.append(filtered)
    docs = list(dedupe.dedupe_stream(iter(raw_docs)))

    import json
    out_path = out / "train.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info("Wrote %d documents to %s", len(docs), out_path)
    return str(out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    run_data_preparation(config_path=args.config, output_dir=args.output_dir)
