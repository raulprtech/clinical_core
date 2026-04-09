"""
TABULAR-CONN Variante A: Cox-PH Baseline
========================================

Direct Cox-PH on raw features. Establishing the C-index floor.
"""

import torch
import numpy as np
from typing import Tuple

class VariantA_CoxBaseline:
    """
    Variant A: Direct Cox-PH on raw features.
    'Encoder' is just z-score normalization + zero-padding to target dim.
    Establishes the C-index floor.
    """
    name = "A_cox_baseline"
    
    def __init__(self, input_dim: int, output_dim: int = 768):
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def encode(
        self, features: np.ndarray, confidence: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        'Encode' by zero-padding raw features to output_dim.
        This is the minimal transformation that satisfies the contract.
        """
        batch_size = features.shape[0]
        
        # Pad to output_dim
        if self.input_dim < self.output_dim:
            padding = np.zeros((batch_size, self.output_dim - self.input_dim))
            embedding = np.concatenate([features, padding], axis=1)
        else:
            embedding = features[:, :self.output_dim]
        
        # L2 normalize
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embedding = embedding / norms
        
        return (
            torch.tensor(embedding, dtype=torch.float32),
            torch.tensor(confidence, dtype=torch.float32).unsqueeze(-1)
        )
