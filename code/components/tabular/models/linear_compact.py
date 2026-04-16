"""
TABULAR-CONN Variante C: Linear Compact Encoder
================================================
    
Compact architecture for edge deployment.
Architecture: input_dim → hidden_dim → output_dim
"""

import torch
import torch.nn as nn
from typing import Tuple

class VariantC_LinearEncoder(nn.Module):
    """
    Variant C: Compact linear encoder.
    
    Architecture: input_dim → hidden_dim → output_dim
    Activation: ReLU (synthesizable as max(0, x))
    Normalization: LayerNorm (post-embedding)
    """
    name = "C_linear_compact"
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 768):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
        # Initialize weights (Xavier for stable gradients)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
    
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing embedding + confidence.
        
        Args:
            x: (batch, input_dim) — preprocessed clinical features
            mask: (batch, input_dim) — 1 if present, 0 if imputed
        
        Returns:
            embedding: (batch, output_dim) — L2-normalized
            confidence: (batch, 1) — weighted completeness score
        """
        # Confidence: weighted fraction of present values
        confidence = mask.float().mean(dim=1, keepdim=True)
        
        # Encoder
        h = self.activation(self.layer1(x))
        embedding = self.norm(self.layer2(h))
        
        # L2 normalize for contract compliance
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        
        return embedding, confidence
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def estimate_flops(self, batch_size: int = 1) -> int:
        """Estimate FLOPs for one forward pass."""
        # Linear layer FLOPs: 2 * in * out (multiply + add)
        flops_l1 = 2 * self.input_dim * self.hidden_dim * batch_size
        flops_relu = self.hidden_dim * batch_size
        flops_l2 = 2 * self.hidden_dim * self.output_dim * batch_size
        flops_norm = 4 * self.output_dim * batch_size  # mean, var, normalize, scale
        return flops_l1 + flops_relu + flops_l2 + flops_norm
