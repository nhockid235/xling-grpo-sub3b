"""Pytest cho YAML config loader (extends:) + JSONL helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.utils.io import load_config, read_jsonl, write_jsonl


def test_load_config_simple(tmp_path: Path):
    cfg = tmp_path / "test.yaml"
    cfg.write_text("foo: 1\nbar: hello\n")
    out = load_config(cfg)
    assert out == {"foo": 1, "bar": "hello"}


def test_load_config_extends(tmp_path: Path):
    parent = tmp_path / "parent.yaml"
    parent.write_text("a: 1\nb:\n  c: 2\n  d: 3\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: parent.yaml\nb:\n  c: 99\ne: 4\n")

    out = load_config(child)
    assert out["a"] == 1
    assert out["b"]["c"] == 99   # overridden
    assert out["b"]["d"] == 3    # inherited
    assert out["e"] == 4         # new
    assert "extends" not in out


def test_jsonl_roundtrip(tmp_path: Path):
    records = [{"id": 1, "text": "hello"}, {"id": 2, "text": "tiếng Việt"}]
    p = tmp_path / "out.jsonl"
    write_jsonl(records, p)

    loaded = read_jsonl(p)
    assert loaded == records


def test_jsonl_unicode(tmp_path: Path):
    """ensure_ascii=False để giữ tiếng Việt nguyên dạng."""
    records = [{"text": "Tính tổng 2 + 2 = ?"}]
    p = tmp_path / "vi.jsonl"
    write_jsonl(records, p)

    raw = p.read_text(encoding="utf-8")
    assert "Tính" in raw   # unicode preserved, not escaped
