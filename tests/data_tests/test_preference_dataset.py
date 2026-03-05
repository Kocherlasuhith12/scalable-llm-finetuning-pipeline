"""Tests for preference dataset (DPO format)."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.datasets.preference_dataset import PreferenceDataset


def test_preference_dataset_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"prompt": "Q?", "chosen": "Good.", "rejected": "Bad."}) + "\n")
        f.write(json.dumps({"instruction": "Tell me", "chosen_response": "Okay.", "rejected_response": "No."}) + "\n")
        path = f.name
    try:
        ds = PreferenceDataset(data_path=path)
        assert len(ds) == 2
        a = ds[0]
        assert a["prompt"] == "Q?"
        assert a["chosen"] == "Good."
        assert a["rejected"] == "Bad."
        b = ds[1]
        assert b["prompt"] == "Tell me"
        assert b["chosen"] == "Okay."
        assert b["rejected"] == "No."
        trl = ds.to_trl_format()
        assert len(trl) == 2
        assert trl[0]["prompt"] == "Q?"
    finally:
        Path(path).unlink(missing_ok=True)
