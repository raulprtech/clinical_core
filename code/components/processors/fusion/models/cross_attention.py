"""Compact trimodal fusion models for small survival cohorts.

The two models in this module share modality-specific projections and a
linear Cox risk head. ``ProjectedConcatSurvivalFusion`` is the parameter-
controlled concatenation reference. ``CrossAttentionSurvivalFusion`` adds
cross-modal self-attention, a small feed-forward block and patient-specific
attention pooling across the three modality tokens.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ModalityTokenEncoder(nn.Module):
    def __init__(
        self,
        modality_dims: Sequence[int],
        d_model: int = 32,
        dropout: float = 0.10,
    ):
        super().__init__()
        if len(modality_dims) < 2:
            raise ValueError("At least two modalities are required")
        self.modality_dims = tuple(int(dim) for dim in modality_dims)
        self.d_model = int(d_model)
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(dim, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
                nn.Dropout(dropout),
            )
            for dim in self.modality_dims
        )
        self.modality_embedding = nn.Parameter(
            torch.empty(len(self.modality_dims), self.d_model)
        )
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)

    def forward(self, modality_inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(modality_inputs) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} modality tensors, "
                f"received {len(modality_inputs)}"
            )
        tokens = []
        batch_size = modality_inputs[0].shape[0]
        for index, (values, projection, expected_dim) in enumerate(
            zip(modality_inputs, self.projections, self.modality_dims)
        ):
            if values.ndim != 2 or values.shape != (batch_size, expected_dim):
                raise ValueError(
                    f"Modality {index} has shape {tuple(values.shape)}; "
                    f"expected ({batch_size}, {expected_dim})"
                )
            tokens.append(projection(values) + self.modality_embedding[index])
        return torch.stack(tokens, dim=1)


class ProjectedConcatSurvivalFusion(nn.Module):
    """Projected-token concatenation followed by a linear Cox head."""

    name = "fusion_projected_concat_survival"

    def __init__(
        self,
        modality_dims: Sequence[int],
        d_model: int = 32,
        dropout: float = 0.10,
    ):
        super().__init__()
        dims = tuple(int(dim) for dim in modality_dims)
        self.encoder = ModalityTokenEncoder(dims, d_model, dropout)
        self.risk_head = nn.Linear(len(dims) * d_model, 1)
        nn.init.xavier_uniform_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)

    def forward(self, modality_inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        tokens = self.encoder(modality_inputs)
        return self.risk_head(tokens.flatten(start_dim=1)).squeeze(-1)

    def modality_weights(self, modality_inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        batch_size = modality_inputs[0].shape[0]
        n_modalities = len(self.encoder.modality_dims)
        return torch.full(
            (batch_size, n_modalities),
            1.0 / n_modalities,
            device=modality_inputs[0].device,
        )


class CrossAttentionSurvivalFusion(nn.Module):
    """Cross-modal attention over one compact token per modality."""

    name = "fusion_cross_attention_survival"

    def __init__(
        self,
        modality_dims: Sequence[int],
        d_model: int = 32,
        num_heads: int = 4,
        dropout: float = 0.10,
        ff_multiplier: int = 2,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.encoder = ModalityTokenEncoder(modality_dims, d_model, dropout)
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(d_model)
        hidden = d_model * int(ff_multiplier)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.pool_score = nn.Linear(d_model, 1)
        self.risk_head = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)

    def attended_tokens(
        self, modality_inputs: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encoder(modality_inputs)
        attended, pairwise_attention = self.cross_attention(
            tokens, tokens, tokens, need_weights=True, average_attn_weights=False
        )
        tokens = self.attention_norm(tokens + attended)
        tokens = self.feed_forward_norm(tokens + self.feed_forward(tokens))
        return tokens, pairwise_attention

    def forward(self, modality_inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        tokens, _ = self.attended_tokens(modality_inputs)
        weights = torch.softmax(self.pool_score(tokens).squeeze(-1), dim=1)
        fused = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return self.risk_head(fused).squeeze(-1)

    @torch.no_grad()
    def modality_weights(self, modality_inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        was_training = self.training
        self.eval()
        tokens, _ = self.attended_tokens(modality_inputs)
        weights = torch.softmax(self.pool_score(tokens).squeeze(-1), dim=1)
        self.train(was_training)
        return weights

    @torch.no_grad()
    def pairwise_attention(
        self, modality_inputs: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        _, attention = self.attended_tokens(modality_inputs)
        self.train(was_training)
        return attention


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
