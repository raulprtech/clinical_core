"""
FUSION-PROC: Concatenation Strategy
===================================

Baseline implementation: confidence-weighted concatenation.
"""

from typing import Dict, Tuple, List
import torch
import numpy as np

class FusionProc_Concatenation:
    """Baseline FUSION-PROC: confidence-weighted concatenation."""
    name = "fusion_baseline_concat"
    
    def __init__(self, modalities: List[str], modality_dims: Dict[str, int] = None, **kwargs):
        self.modalities = modalities
        self.modality_dims = modality_dims or {m: 768 for m in modalities}
        self.fused_dim = sum(self.modality_dims[m] for m in modalities) + len(modalities)
        self._weight_by_confidence = kwargs.get('weight_by_confidence', True)
    
    def fuse_one(self, modality_outputs: Dict[str, Tuple[torch.Tensor, float]]) -> Tuple[torch.Tensor, float]:
        parts = []
        confidences = []
        for mod_name in self.modalities:
            dim = self.modality_dims[mod_name]
            if mod_name in modality_outputs:
                emb, conf = modality_outputs[mod_name]
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb, dtype=torch.float32)
                if self._weight_by_confidence:
                    emb = emb * conf
            else:
                emb = torch.zeros(dim); conf = 0.0
            parts.append(emb); confidences.append(conf)
        embeddings_concat = torch.cat(parts, dim=-1)
        conf_tensor = torch.tensor(confidences, dtype=torch.float32)
        fused = torch.cat([embeddings_concat, conf_tensor], dim=-1)
        present = [c for c in confidences if c > 0.0]
        aggregate_conf = float(np.mean(present)) if present else 0.0
        return fused, aggregate_conf
    
    def fuse_batch(self, batch_outputs: List[Dict[str, Tuple[torch.Tensor, float]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        fused_list, conf_list = [], []
        for patient_outputs in batch_outputs:
            fused, conf = self.fuse_one(patient_outputs)
            fused_list.append(fused); conf_list.append(conf)
        return torch.stack(fused_list), torch.tensor(conf_list, dtype=torch.float32)
