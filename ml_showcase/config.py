from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a Python dict."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Config:
    seed: int
    device: str
    data: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    causal: dict[str, Any]


def load_config(path: str | Path) -> Config:
    d = load_yaml(path)
    return Config(
        seed=int(d.get("seed", 123)),
        device=str(d.get("device", "cpu")),
        data=dict(d.get("data", {})),
        model=dict(d.get("model", {})),
        training=dict(d.get("training", {})),
        causal=dict(d.get("causal", {})),
    )
