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
            padding = np.zeros(
                (batch_size, self.output_dim - self.input_dim),
                dtype=np.float32,
            )
            embedding = np.concatenate([features, padding], axis=1)
        else:
            embedding = features[:, :self.output_dim]

        # The ingestion contract requires unit L2 norm. Cox predictions do not
        # use this projection; normalization only makes the reusable embedding
        # compatible with downstream connectors.
        embedding = embedding.astype(np.float32, copy=False)
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        zero_rows = norms[:, 0] == 0
        if np.any(zero_rows):
            embedding = embedding.copy()
            embedding[zero_rows, 0] = 1.0
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / norms
        
        return (
            torch.from_numpy(embedding),
            torch.tensor(confidence, dtype=torch.float32).unsqueeze(-1)
        )
