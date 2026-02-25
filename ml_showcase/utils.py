from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    metrics_path: Path
    causal_path: Path


def make_run_dir(base_dir: str = "runs") -> RunPaths:
    ts = np.datetime64("now").astype(str).replace(":", "").replace("T", "_")
    run_dir = ensure_dir(Path(base_dir) / ts)
    return RunPaths(
        run_dir=run_dir,
        metrics_path=run_dir / "metrics.json",
        causal_path=run_dir / "causal_edges.csv",
    )


def save_json(obj: dict[str, Any], path: str | os.PathLike[str]) -> None:
    import json

    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
