from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CausalEdges:
    """Edge table returned by causal discovery."""

    edges: pd.DataFrame  # columns: src, dst, weight, freq


def _partial_corr(X: np.ndarray) -> np.ndarray:
    """Estimate partial correlation via inverse covariance (precision matrix)."""
    X = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(X, rowvar=False)

    # Regularize for numerical stability
    cov = cov + np.eye(cov.shape[0]) * 1e-3
    prec = np.linalg.pinv(cov)

    d = np.sqrt(np.clip(np.diag(prec), 1e-12, None))
    pc = -prec / (d[:, None] * d[None, :])
    np.fill_diagonal(pc, 0.0)
    return pc


def bootstrap_edges(
    modules: np.ndarray, n_boot: int, edge_threshold: float, seed: int
) -> CausalEdges:
    """Bootstrap partial-correlation edges across resampled rows."""
    rng = np.random.default_rng(seed)
    n, p = modules.shape

    counts = np.zeros((p, p), dtype=np.int64)
    weights = np.zeros((p, p), dtype=np.float32)

    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        pc = _partial_corr(modules[idx])
        sel = np.abs(pc) >= edge_threshold
        counts += sel.astype(np.int64)
        weights += pc.astype(np.float32)

    freq = counts / max(1, int(n_boot))
    avg = weights / max(1, int(n_boot))

    rows: list[tuple[int, int, float, float]] = []
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if float(freq[i, j]) >= 0.6:
                rows.append((i, j, float(avg[i, j]), float(freq[i, j])))

    df = pd.DataFrame(rows, columns=["src", "dst", "weight", "freq"]).sort_values(
        ["freq", "weight"], ascending=[False, False]
    )
    return CausalEdges(edges=df)
