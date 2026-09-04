"""Terminal complexity prediction Transformer.

Input tokens (37 variable tokens + 1 global token = 38):
    BagDrop (13) + DG_QL (8) + DG_WT (8) + SC_QL (4) + SC_WT (4)
"""

import torch
import torch.nn as nn

from encoder import TransformerEncoderLayerWithAttention


class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_ff, dropout, max_len_pe: int = 200):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # fixed token-group sizes for this task
        self.num_bagdrop = 13
        self.num_dg = 8
        self.num_sc = 4
        self.total_tokens = self.num_bagdrop + 2 * self.num_dg + 2 * self.num_sc  # 37

        # value projection: each variable has an 18-step history -> (B, N_var, 18)
        self.input_embedding = nn.Linear(18, d_model)

        # positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_pe, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # global (CLS-style) token
        self.global_context = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.global_context, std=0.02)

        # cyclic hour/day time context -> MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(4, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.time_scale = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))

        # type embedding: 0=BagDrop, 1=DG_QL, 2=DG_WT, 3=SC_QL, 4=SC_WT
        self.num_types = 5
        self.type_embedding = nn.Embedding(self.num_types, d_model)

        # source embedding: facility identity
        # BagDrop -> 0~12, DG -> 13~20 (shared by DG_QL/DG_WT), SC -> 21~24 (shared by SC_QL/SC_WT)
        self.num_sources = 25
        self.source_embedding = nn.Embedding(self.num_sources, d_model)

        self.type_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.source_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithAttention(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                activation="gelu",
                norm_first=True,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # pooling fusion: global + avg + max
        self.pool_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # output heads: DG = 2 x 8 x 12, SC = 2 x 4 x 12
        self.departure_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 96 * 2),
        )
        self.security_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 48 * 2),
        )

        type_ids, source_ids = self._build_token_id_tables()
        self.register_buffer("token_type_ids", type_ids, persistent=False)      # (37,)
        self.register_buffer("token_source_ids", source_ids, persistent=False)  # (37,)

    def _build_token_id_tables(self):
        """Token order: BagDrop(13), DG_QL(8), DG_WT(8), SC_QL(4), SC_WT(4)."""
        type_ids, source_ids = [], []

        for i in range(self.num_bagdrop):          # BagDrop: type=0, source=0~12
            type_ids.append(0)
            source_ids.append(i)
        for i in range(self.num_dg):                # DG_QL: type=1, source=13~20
            type_ids.append(1)
            source_ids.append(13 + i)
        for i in range(self.num_dg):                # DG_WT: type=2, source=13~20 (shared)
            type_ids.append(2)
            source_ids.append(13 + i)
        for i in range(self.num_sc):                # SC_QL: type=3, source=21~24
            type_ids.append(3)
            source_ids.append(21 + i)
        for i in range(self.num_sc):                # SC_WT: type=4, source=21~24 (shared)
            type_ids.append(4)
            source_ids.append(21 + i)

        return torch.tensor(type_ids, dtype=torch.long), torch.tensor(source_ids, dtype=torch.long)

    def _make_time_context(self, Day, Hour):
        """Day: (B,) in [0, 6], Hour: (B,) in [0, 17]."""
        h_norm = Hour.float() / 17.0
        d_norm = Day.float() / 6.0

        h_sin = torch.sin(2 * torch.pi * h_norm).unsqueeze(-1)
        h_cos = torch.cos(2 * torch.pi * h_norm).unsqueeze(-1)
        d_sin = torch.sin(2 * torch.pi * d_norm).unsqueeze(-1)
        d_cos = torch.cos(2 * torch.pi * d_norm).unsqueeze(-1)

        time_feat = torch.cat([h_sin, h_cos, d_sin, d_cos], dim=-1).float()  # (B, 4)
        return self.time_mlp(time_feat).unsqueeze(1)  # (B, 1, d_model)

    def _embed_variable_group(self, x_group, type_id_slice, source_id_slice):
        """x_group: (B, N_group, 18) -> (B, N_group, d_model)."""
        B, N, _ = x_group.shape

        value_emb = self.input_embedding(x_group)

        type_ids = self.token_type_ids[type_id_slice]
        source_ids = self.token_source_ids[source_id_slice]

        type_emb = self.type_embedding(type_ids).unsqueeze(0).expand(B, -1, -1)
        source_emb = self.source_embedding(source_ids).unsqueeze(0).expand(B, -1, -1)

        return value_emb + self.type_scale * type_emb + self.source_scale * source_emb

    def forward(self, x, return_attn=False):
        (
            BagDrop,         # (B, 13, 18)
            DG_QL_before,    # (B,  8, 18)
            DG_WT_before,    # (B,  8, 18)
            SC_QL_before,    # (B,  4, 18)
            SC_WT_before,    # (B,  4, 18)
            Day,             # (B,) or (B, 1)
            Hour,            # (B,) or (B, 1)
        ) = x

        B = BagDrop.shape[0]
        if Day.dim() > 1:
            Day = Day[:, 0]
        if Hour.dim() > 1:
            Hour = Hour[:, 0]

        # value + type + source embeddings, token order: [0:13, 13:21, 21:29, 29:33, 33:37]
        BagDrop_emb = self._embed_variable_group(BagDrop,      slice(0, 13),  slice(0, 13))
        DG_QL_emb   = self._embed_variable_group(DG_QL_before, slice(13, 21), slice(13, 21))
        DG_WT_emb   = self._embed_variable_group(DG_WT_before, slice(21, 29), slice(21, 29))
        SC_QL_emb   = self._embed_variable_group(SC_QL_before, slice(29, 33), slice(29, 33))
        SC_WT_emb   = self._embed_variable_group(SC_WT_before, slice(33, 37), slice(33, 37))

        input_tokens = torch.cat(
            [BagDrop_emb, DG_QL_emb, DG_WT_emb, SC_QL_emb, SC_WT_emb], dim=1
        )  # (B, 37, d_model)

        # time context is broadcast to all variable tokens
        time_ctx = self._make_time_context(Day, Hour)
        input_tokens = input_tokens + self.time_scale * time_ctx

        # prepend global token + positional embedding
        global_token = self.global_context.expand(B, -1, -1)
        inputs = torch.cat([global_token, input_tokens], dim=1)  # (B, 38, d_model)

        seq_len = inputs.shape[1]
        if seq_len > self.pos_emb.size(1):
            raise ValueError(f"seq_len={seq_len} exceeds max_len_pe={self.pos_emb.size(1)}")
        x = inputs + self.pos_emb[:, :seq_len, :]

        attn_list = []
        for layer in self.encoder_layers:
            x = layer(x)
            if return_attn:
                attn_list.append(layer.last_attn_weights)

        # pooling: global token + avg + max over variable tokens
        global_info = x[:, 0, :]
        seq_tokens = x[:, 1:, :]
        avg_pool = seq_tokens.mean(dim=1)
        max_pool = seq_tokens.max(dim=1).values

        pooled = torch.cat([global_info, avg_pool, max_pool], dim=-1)
        pooled_info = self.final_norm(self.pool_fusion(pooled))

        # output heads: DG = 2 x 8 x 12, SC = 2 x 4 x 12
        departure_pred = self.departure_projection(pooled_info).contiguous().view(B, 2, 8, 12)
        security_pred = self.security_projection(pooled_info).contiguous().view(B, 2, 4, 12)

        departure_QL = departure_pred[:, 0]
        departure_WT = departure_pred[:, 1]
        security_QL = security_pred[:, 0]
        security_WT = security_pred[:, 1]

        if return_attn:
            return departure_QL, departure_WT, security_QL, security_WT, attn_list
        return departure_QL, departure_WT, security_QL, security_WT
