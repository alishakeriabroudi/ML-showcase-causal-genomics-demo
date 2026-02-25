from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticBatch:
    """Ragged set inputs + labels + latent module activations."""

    mut_idx: np.ndarray
    offsets: np.ndarray
    y: np.ndarray
    modules: np.ndarray


def make_synthetic(
    *,
    n_samples: int,
    vocab_size: int,
    max_set_len: int,
    n_modules: int,
    shift_strength: float,
    seed: int,
) -> SyntheticBatch:
    """Generate a simple synthetic dataset."""
    rng = np.random.default_rng(seed)

    W = rng.normal(0.0, 1.0, size=(vocab_size, n_modules)).astype(np.float32)
    if shift_strength > 0:
        W = W + rng.normal(0.0, shift_strength, size=W.shape).astype(np.float32)

    lengths = rng.integers(low=1, high=max(2, max_set_len + 1), size=n_samples, dtype=np.int64)
    offsets = np.zeros(n_samples + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)

    mut_idx = rng.integers(0, vocab_size, size=int(offsets[-1]), dtype=np.int64)

    modules = np.zeros((n_samples, n_modules), dtype=np.float32)
    for i in range(n_samples):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        toks = mut_idx[s:e]
        modules[i] = W[toks].mean(axis=0)

    beta = rng.normal(0.0, 1.0, size=(n_modules,)).astype(np.float32)
    logits = modules @ beta + rng.normal(0.0, 0.2, size=(n_samples,)).astype(np.float32)
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n_samples) < prob).astype(np.float32)

    return SyntheticBatch(mut_idx=mut_idx, offsets=offsets, y=y, modules=modules)
