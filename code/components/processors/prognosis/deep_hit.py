"""
PROGNOSIS-PROC: DeepHit Strategy
================================

(Futura expansión)
Reference: Lee et al. (2018) AAAI.
"""

import torch
import torch.nn as nn

class PrognosisProc_DeepHit(nn.Module):
    """Futura implementación de DeepHit (PMF de tiempo discreto)."""
    name = "prognosis_deephit"
    def __init__(self, **kwargs):
        super().__init__()
        pass
    def forward(self, x):
        return x
