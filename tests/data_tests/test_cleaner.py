"""Tests for text cleaner."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.processors.cleaner import TextCleaner


def test_cleaner_basic():
    c = TextCleaner(min_length=2, remove_urls=True)
    assert c.clean("  Hello world  ") == "Hello world"
    assert c.clean("See https://example.com for more") == "See for more"
    assert c.clean("a") == ""


def test_clean_document():
    c = TextCleaner(min_length=2)
    doc = {"id": "1", "text": "  Good content here  "}
    out = c.clean_document(doc)
    assert out is not None
    assert out["text"] == "Good content here"
    assert c.clean_document({"id": "2", "text": "x"}) is None
