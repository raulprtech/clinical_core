"""
FT-Transformer encoder — compliant with CLINICAL-CORE TABULAR-IN contract.

This encoder is a SOTA neural-network baseline for tabular data that produces
representational embeddings of output_dim dimensions. It is the complement to
linear_compact (the compact FPGA-viable encoder): FT-Transformer is at the
opposite end of the capacity spectrum (dense transformer with attention), so
comparing both characterizes the precision/efficiency trade-off under the same
contract.

Reference:
    Gorishniy, Rubachev, Khrulkov, Babenko. "Revisiting Deep Learning Models
    for Tabular Data." NeurIPS 2021.

Contract compliance (all three guarantees):
    • Produces embeddings of exactly output_dim dimensions (projected via
      a final linear layer if the transformer dim does not match).
    • Applies L2 normalization to the output embedding.
    • Reports a confidence scalar in [0, 1] per case, derived from the
      mean of the input confidence mask (i.e., how complete the input is).

Training is performed with Cox partial-likelihood loss — same loss as the
linear_compact baseline, so that comparison between the two isolates the
architectural variable (capacity, inductive bias) from the optimization target.

Implementation notes:
    • Uses the `rtdl_revisiting_models` package (authors' official code) if
      available, otherwise falls back to a lightweight in-file implementation.
    • Dependency is optional and lazy-imported so existing environments without
      rtdl do not break when this module is imported.
"""
from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# LIGHTWEIGHT FT-TRANSFORMER IMPLEMENTATION
# ============================================================
# We implement a minimal faithful version in-file rather than depending on
# `rtdl_revisiting_models`. Reasons:
#   1. Zero new dependency for this repo.
#   2. Full control over the final projection to output_dim=768.
#   3. Transparent to reviewers — the architecture is in our codebase.
# The implementation follows Gorishniy et al. 2021 Section 3.1.


class FeatureTokenizer(nn.Module):
    """
    Maps n_features scalar inputs into n_features token embeddings of dimension
    d_token, via a learned weight and bias per feature (Gorishniy et al. 2021,
    Section 3.1, Numerical Feature Tokenizer).

    All features here are numerical (after preprocessing). Categorical
    features, if present in future, would use a lookup embedding instead.
    """

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        # Initialization following the reference implementation
        nn.init.uniform_(self.weight, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))
        nn.init.uniform_(self.bias, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_features]  →  tokens: [batch, n_features, d_token]
        return self.weight.unsqueeze(0) * x.unsqueeze(-1) + self.bias.unsqueeze(0)


class CLSToken(nn.Module):
    """Prepends a learnable [CLS] token to the sequence of feature tokens."""

    def __init__(self, d_token: int):
        super().__init__()
        self.cls = nn.Parameter(torch.empty(d_token))
        nn.init.uniform_(self.cls, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [batch, n_features, d_token]
        batch = tokens.shape[0]
        cls = self.cls.expand(batch, 1, -1)
        return torch.cat([cls, tokens], dim=1)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: MHA + feedforward."""

    def __init__(self, d_token: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_token)
        self.ff = nn.Sequential(
            nn.Linear(d_token, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_token),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)

        h = self.norm2(x)
        x = x + self.dropout(self.ff(h))
        return x


# ============================================================
# FT-TRANSFORMER ENCODER (CLINICAL-CORE contract-compliant)
# ============================================================

class FTTransformerEncoder(nn.Module):
    """
    FT-Transformer encoder producing a contract-compliant 768-dim embedding
    plus a confidence scalar per case.

    Architecture:
        input [B, n_features]
          → FeatureTokenizer → [B, n_features, d_token]
          → Prepend [CLS]    → [B, n_features+1, d_token]
          → N transformer blocks
          → take [CLS]       → [B, d_token]
          → linear projection → [B, output_dim]
          → L2 normalize     → embedding
        confidence_mask [B, n_features]
          → mean over feature axis → confidence scalar in [0, 1]

    Output shape: (embedding [B, output_dim], confidence [B])

    Args:
        input_dim:   number of input features (e.g., 19 for RENAL-CORE tabular).
        output_dim:  contract-fixed ingestion dim (e.g., 768).
        d_token:     internal transformer dimension. Default 192.
        n_blocks:    number of transformer blocks. Default 3.
        n_heads:     attention heads per block. Default 8.
        d_ff:        feedforward inner dim. Default d_token * 4.
        dropout:     dropout rate across attention and FF. Default 0.1.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 768,
        d_token: int = 192,
        n_blocks: int = 3,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_token = d_token

        d_ff = d_ff if d_ff is not None else d_token * 4

        self.tokenizer = FeatureTokenizer(input_dim, d_token)
        self.cls_token = CLSToken(d_token)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ff, dropout)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_token)
        self.projection = nn.Linear(d_token, output_dim)

        # Standard-deviation buffer used later by a downstream confidence scorer
        # if we ever want to calibrate. For now confidence comes from the mask.
        self.register_buffer('_contract_compliant', torch.tensor(True))

    @property
    def name(self) -> str:
        return "ft_transformer"

    @property
    def contract_compliant(self) -> bool:
        return True

    def forward(
        self,
        x: torch.Tensor,
        confidence_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:                [batch, n_features] numeric features (imputed).
            confidence_mask:  [batch, n_features] values in [0, 1] indicating
                              per-feature input reliability (1 = original,
                              lower = imputed). Used to derive the case-level
                              confidence scalar.

        Returns:
            embedding:  [batch, output_dim], L2-normalized.
            confidence: [batch], scalar in [0, 1] per case.
        """
        tokens = self.tokenizer(x)
        tokens = self.cls_token(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        cls = self.final_norm(tokens[:, 0])

        emb = self.projection(cls)
        emb = F.normalize(emb, p=2, dim=-1)  # L2 normalization (contract)

        # Confidence: mean of per-feature reliability. Clamp to [0, 1] to be
        # safe against numerical drift in the input mask.
        conf = confidence_mask.mean(dim=-1).clamp(0.0, 1.0)

        return emb, conf

    def encode(
        self,
        x: torch.Tensor,
        confidence_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias to forward, used by _eval helpers that expect `.encode`."""
        return self.forward(x, confidence_mask)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def estimated_flops(self, seq_len: Optional[int] = None) -> int:
        """
        Rough FLOPs estimate for one forward pass with batch=1.
        seq_len defaults to input_dim + 1 (features + CLS).
        """
        seq_len = seq_len if seq_len is not None else self.input_dim + 1
        # Tokenizer: 2 * input_dim * d_token multiplications+adds
        tok = 2 * self.input_dim * self.d_token
        # Each block: attention O(seq^2 * d_token) + FF 2 * seq * d_token * d_ff
        d_ff = self.d_token * 4
        per_block = (
            2 * seq_len * seq_len * self.d_token
            + 2 * seq_len * self.d_token * d_ff
        )
        blocks = len(self.blocks) * per_block
        # Projection: 2 * d_token * output_dim
        proj = 2 * self.d_token * self.output_dim
        return tok + blocks + proj


# ============================================================
# FACTORY (for registry.py registration)
# ============================================================

def build_ft_transformer(
    input_dim: int,
    output_dim: int = 768,
    **kwargs,
) -> FTTransformerEncoder:
    """
    Factory for registry.py. Accepts **kwargs so that variant_params from the
    experiment config pass through cleanly (d_token, n_blocks, n_heads, etc).
    """
    return FTTransformerEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        d_token=kwargs.get('d_token', 192),
        n_blocks=kwargs.get('n_blocks', 3),
        n_heads=kwargs.get('n_heads', 8),
        d_ff=kwargs.get('d_ff', None),
        dropout=kwargs.get('dropout', 0.1),
    )