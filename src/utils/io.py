"""IO utilities — YAML config loader (with `extends:` support), JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config, resolving `extends:` directive (1 level deep).

    Child config keys override parent. Nested dicts deep-merged.
    """
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f)

    if "extends" in cfg:
        parent_path = path.parent / cfg.pop("extends")
        parent = load_config(parent_path)
        cfg = _deep_merge(parent, cfg)

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
