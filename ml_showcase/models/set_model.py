from __future__ import annotations

import torch
from torch import nn


class AttnPool(nn.Module):
    """Lightweight attention pooling over a ragged set."""

    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.q = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.q, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        q = self.q.expand(B, -1, -1)
        pad = ~mask
        out, _ = self.attn(q, x, x, key_padding_mask=pad, need_weights=False)
        return out.squeeze(1)


class Predictor(nn.Module):
    """Set encoder + binary classifier."""

    def __init__(
        self, vocab_size: int, embed_dim: int, set_hidden: int, attn_heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.phi = nn.Sequential(
            nn.Linear(embed_dim, set_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(set_hidden, embed_dim),
        )
        self.pool = AttnPool(dim=embed_dim, n_heads=attn_heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, set_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(set_hidden, 1),
        )

    def forward(self, mut_idx: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        B = offsets.numel() - 1
        lengths = (offsets[1:] - offsets[:-1]).tolist()
        max_len = int(max(lengths)) if lengths else 0

        x = mut_idx.new_zeros((B, max_len), dtype=torch.long)
        mask = torch.zeros((B, max_len), dtype=torch.bool, device=mut_idx.device)

        for i in range(B):
            s = int(offsets[i].item())
            e = int(offsets[i + 1].item())
            n = e - s
            if n <= 0:
                continue
            x[i, :n] = mut_idx[s:e]
            mask[i, :n] = True

        z = self.emb(x)
        z = self.phi(z)
        pooled = self.pool(z, mask=mask)
        logits = self.head(pooled).squeeze(-1)
        return logits
