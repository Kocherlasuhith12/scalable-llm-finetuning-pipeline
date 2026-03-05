"""Tests for deduplicator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.validators.deduplicator import Deduplicator


def test_dedupe_exact():
    dedupe = Deduplicator()
    docs = [
        {"text": "Same content"},
        {"text": "Same content"},
        {"text": "Other content"},
    ]
    out = list(dedupe.dedupe_stream(iter(docs)))
    assert len(out) == 2
    assert out[0]["text"] == "Same content"
    assert out[1]["text"] == "Other content"
