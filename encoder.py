"""Pre-norm Transformer encoder layer that exposes its attention weights."""

import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayerWithAttention(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=1024,
        dropout=0.1,
        activation="gelu",
        norm_first=True,
        batch_first=True,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.last_attn_weights = None  # (B, H, T, T)

    def _sa_block(self, x):
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attn_weights = attn_weights
        return self.dropout1(attn_out)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, x):
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))
        return x
