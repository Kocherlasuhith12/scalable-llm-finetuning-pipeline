"""Integration test for config loading."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_parser import load_config, merge_configs


def test_merge_configs():
    base = {"a": 1, "b": {"c": 2}}
    override = {"b": {"c": 3}, "d": 4}
    out = merge_configs(base, override)
    assert out["a"] == 1
    assert out["b"]["c"] == 3
    assert out["d"] == 4


def test_load_config_with_base():
    with tempfile.TemporaryDirectory() as tmp:
        base_path = Path(tmp) / "base.yaml"
        base_path.write_text("x: 1\ny: 2\n")
        child_path = Path(tmp) / "child.yaml"
        child_path.write_text("_base: base.yaml\ny: 3\nz: 4\n")
        out = load_config(child_path)
        assert out["x"] == 1
        assert out["y"] == 3
        assert out["z"] == 4
