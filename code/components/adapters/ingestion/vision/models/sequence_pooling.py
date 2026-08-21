"""Lightweight patient-level pooling for frozen ResNet18 2.5D sequences.

The attention and Mamba variants share the same token projector, positional
features, gated pooling layer and Cox objective. The only architectural
difference is the selective state-space stack inserted by the Mamba variant.
This makes attention a useful parameter-aware control for longitudinal
sequence modelling.

The selective scan is written in plain PyTorch. It follows the core Mamba
parameterization (input-dependent B, C and delta with diagonal state matrix)
without requiring the optional CUDA-only fused kernels from mamba-ssm.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_sequence_inputs(
    features: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if features.ndim != 3:
        raise ValueError(f"features must have shape [B, T, F], got {features.shape}")
    if positions.shape != features.shape[:2]:
        raise ValueError("positions must have shape [B, T]")
    if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean with shape [B, T]")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every patient must contain at least one valid token")


class TokenProjector(nn.Module):
    """Project frozen features and inject scanner-independent relative position."""

    def __init__(
        self,
        input_dim: int = 512,
        model_dim: int = 128,
        dropout: float = 0.1,
        use_position: bool = True,
    ):
        super().__init__()
        self.use_position = bool(use_position)
        self.feature_norm = nn.LayerNorm(input_dim)
        self.feature_projection = nn.Linear(input_dim, model_dim)
        self.position_projection = nn.Linear(3, model_dim, bias=False)
        self.output_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        tokens = self.feature_projection(self.feature_norm(features))
        if self.use_position:
            position_features = torch.stack(
                (
                    positions,
                    torch.sin(math.pi * positions),
                    torch.cos(math.pi * positions),
                ),
                dim=-1,
            )
            tokens = tokens + self.position_projection(position_features)
        return self.dropout(self.output_norm(F.gelu(tokens)))


class GatedAttentionPool(nn.Module):
    """Ilse-style gated attention with explicit padding masks."""

    def __init__(self, model_dim: int = 128, attention_dim: int = 64):
        super().__init__()
        self.value = nn.Linear(model_dim, attention_dim)
        self.gate = nn.Linear(model_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(
            torch.tanh(self.value(tokens)) * torch.sigmoid(self.gate(tokens))
        ).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class AttentionSequenceSurvival(nn.Module):
    """ResNet18 tokens -> gated attention pooling -> Cox risk."""

    def __init__(
        self,
        input_dim: int = 512,
        model_dim: int = 128,
        attention_dim: int = 64,
        dropout: float = 0.1,
        use_position: bool = True,
    ):
        super().__init__()
        self.projector = TokenProjector(
            input_dim, model_dim, dropout, use_position=use_position
        )
        self.pool = GatedAttentionPool(model_dim, attention_dim)
        self.risk_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ):
        _validate_sequence_inputs(features, positions, mask)
        tokens = self.projector(features, positions)
        pooled, weights = self.pool(tokens, mask)
        risk = self.risk_head(pooled).squeeze(-1)
        return (risk, weights) if return_attention else risk


class SelectiveStateSpaceBlock(nn.Module):
    """Reference selective SSM block using the core Mamba recurrence."""

    def __init__(
        self,
        model_dim: int = 128,
        state_dim: int = 16,
        conv_width: int = 4,
        expansion: int = 2,
        dt_rank: Optional[int] = None,
    ):
        super().__init__()
        if min(model_dim, state_dim, conv_width, expansion) < 1:
            raise ValueError("Mamba dimensions must be positive")
        self.model_dim = int(model_dim)
        self.state_dim = int(state_dim)
        self.inner_dim = int(expansion) * self.model_dim
        self.dt_rank = int(dt_rank or math.ceil(self.model_dim / 16))
        self.input_projection = nn.Linear(self.model_dim, self.inner_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=int(conv_width),
            padding=int(conv_width) - 1,
            groups=self.inner_dim,
        )
        self.parameter_projection = nn.Linear(
            self.inner_dim, self.dt_rank + self.state_dim * 2, bias=False
        )
        self.delta_projection = nn.Linear(self.dt_rank, self.inner_dim)
        base = torch.arange(1, self.state_dim + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(base.log().repeat(self.inner_dim, 1))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.output_projection = nn.Linear(self.inner_dim, self.model_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tokens.shape
        projected, gate = self.input_projection(tokens).chunk(2, dim=-1)
        convolved = self.depthwise_conv(projected.transpose(1, 2))
        convolved = F.silu(convolved[:, :, :sequence_length].transpose(1, 2))
        parameters = self.parameter_projection(convolved)
        delta_raw, input_B, output_C = torch.split(
            parameters, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        delta = F.softplus(self.delta_projection(delta_raw))
        A = -torch.exp(self.A_log.float()).to(tokens.dtype)
        state = tokens.new_zeros(batch_size, self.inner_dim, self.state_dim)
        outputs = []
        for index in range(sequence_length):
            valid = mask[:, index].view(batch_size, 1, 1)
            delta_t = delta[:, index]
            input_t = convolved[:, index]
            decay = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
            candidate = decay * state + (
                delta_t.unsqueeze(-1)
                * input_B[:, index].unsqueeze(1)
                * input_t.unsqueeze(-1)
            )
            state = torch.where(valid, candidate, state)
            output_t = torch.sum(
                state * output_C[:, index].unsqueeze(1), dim=-1
            ) + self.D * input_t
            output_t = output_t * F.silu(gate[:, index])
            outputs.append(output_t * mask[:, index].unsqueeze(-1))
        output = torch.stack(outputs, dim=1)
        return self.output_projection(output)


class MambaResidualBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        conv_width: int,
        expansion: int,
        dropout: float,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.ssm = SelectiveStateSpaceBlock(
            model_dim=model_dim,
            state_dim=state_dim,
            conv_width=conv_width,
            expansion=expansion,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = tokens + self.dropout(self.ssm(self.norm(tokens), mask))
        return output * mask.unsqueeze(-1)


class MambaSequenceSurvival(nn.Module):
    """ResNet18 tokens -> Mamba blocks -> shared attention pool -> Cox risk."""

    def __init__(
        self,
        input_dim: int = 512,
        model_dim: int = 128,
        attention_dim: int = 64,
        state_dim: int = 16,
        n_blocks: int = 2,
        conv_width: int = 4,
        expansion: int = 2,
        dropout: float = 0.1,
        use_position: bool = True,
    ):
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be positive")
        self.projector = TokenProjector(
            input_dim, model_dim, dropout, use_position=use_position
        )
        self.blocks = nn.ModuleList(
            MambaResidualBlock(
                model_dim, state_dim, conv_width, expansion, dropout
            )
            for _ in range(int(n_blocks))
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.pool = GatedAttentionPool(model_dim, attention_dim)
        self.risk_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ):
        _validate_sequence_inputs(features, positions, mask)
        tokens = self.projector(features, positions) * mask.unsqueeze(-1)
        for block in self.blocks:
            tokens = block(tokens, mask)
        tokens = self.final_norm(tokens) * mask.unsqueeze(-1)
        pooled, weights = self.pool(tokens, mask)
        risk = self.risk_head(pooled).squeeze(-1)
        return (risk, weights) if return_attention else risk


def cox_ph_loss(
    risk: torch.Tensor, survival_days: torch.Tensor, events: torch.Tensor
) -> torch.Tensor:
    """Negative Cox partial log-likelihood using full split risk sets."""
    if not (risk.ndim == survival_days.ndim == events.ndim == 1):
        raise ValueError("risk, survival_days and events must be one-dimensional")
    if not (len(risk) == len(survival_days) == len(events)):
        raise ValueError("Cox inputs must have equal length")
    order = torch.argsort(survival_days, descending=True)
    ordered_risk = risk[order]
    ordered_events = events[order].to(risk.dtype)
    event_count = ordered_events.sum()
    if float(event_count.detach().cpu()) <= 0:
        raise ValueError("Cox loss requires at least one observed event")
    log_risk_set = torch.logcumsumexp(ordered_risk, dim=0)
    return -torch.sum((ordered_risk - log_risk_set) * ordered_events) / event_count


def build_sequence_model(name: str, **kwargs) -> nn.Module:
    if name == "attention":
        return AttentionSequenceSurvival(**kwargs)
    if name == "mamba":
        return MambaSequenceSurvival(**kwargs)
    raise KeyError(f"Unknown sequence model: {name}")
