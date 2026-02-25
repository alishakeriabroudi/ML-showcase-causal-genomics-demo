from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from .models.set_model import Predictor


class RaggedSetDataset(TorchDataset):
    def __init__(self, mut_idx: np.ndarray, offsets: np.ndarray, y: np.ndarray) -> None:
        self.mut_idx = mut_idx.astype(np.int64)
        self.offsets = offsets.astype(np.int64)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return int(self.offsets.size - 1)

    def __getitem__(self, i: int):
        s = int(self.offsets[i])
        e = int(self.offsets[i + 1])
        return self.mut_idx[s:e], self.y[i]


def collate(batch):
    if not batch:
        return [], torch.tensor([], dtype=torch.float32)

    muts, ys = zip(*batch, strict=True)

    offsets = [0]
    flat: list[int] = []
    for m in muts:
        flat.extend(m.tolist())
        offsets.append(offsets[-1] + len(m))

    return (
        torch.tensor(flat, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(ys, dtype=torch.float32),
    )


@dataclass(frozen=True)
class TrainResult:
    auc: float


def train_eval(
    *,
    train_mut_idx: np.ndarray,
    train_offsets: np.ndarray,
    train_y: np.ndarray,
    test_mut_idx: np.ndarray,
    test_offsets: np.ndarray,
    test_y: np.ndarray,
    vocab_size: int,
    embed_dim: int,
    set_hidden: int,
    attn_heads: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    device: str,
) -> TrainResult:
    dev = torch.device(device)
    model = Predictor(vocab_size, embed_dim, set_hidden, attn_heads, dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_ds = RaggedSetDataset(train_mut_idx, train_offsets, train_y)
    test_ds = RaggedSetDataset(test_mut_idx, test_offsets, test_y)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model.train()
    for _ in range(int(epochs)):
        for mut_idx, offsets, y in train_dl:
            mut_idx, offsets, y = mut_idx.to(dev), offsets.to(dev), y.to(dev)

            opt.zero_grad(set_to_none=True)
            logits = model(mut_idx, offsets)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

    model.eval()
    ys: list[float] = []
    ps: list[float] = []
    with torch.no_grad():
        for mut_idx, offsets, y in test_dl:
            mut_idx, offsets = mut_idx.to(dev), offsets.to(dev)

            logits = model(mut_idx, offsets).cpu().numpy()
            prob = 1.0 / (1.0 + np.exp(-logits))
            ys.extend(y.numpy().tolist())
            ps.extend(prob.tolist())

    auc = float(roc_auc_score(ys, ps))
    return TrainResult(auc=auc)
