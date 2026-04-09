"""
PROGNOSIS-PROC: Linear Cox Strategy
====================================

Baseline: linear head trained with Cox partial likelihood loss.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index

def cox_partial_likelihood_loss(risk_scores: torch.Tensor, survival_times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    sorted_idx = torch.argsort(survival_times, descending=True)
    risk_sorted = risk_scores[sorted_idx]
    events_sorted = events[sorted_idx]
    log_cumsum = torch.logcumsumexp(risk_sorted, dim=0)
    uncensored = risk_sorted - log_cumsum
    event_mask = events_sorted.bool()
    if event_mask.sum() == 0: return torch.tensor(0.0, requires_grad=True)
    return -uncensored[event_mask].mean()

class PrognosisProc_LinearCox(nn.Module):
    name = "prognosis_baseline_linear_cox"
    def __init__(self, fused_dim: int, **kwargs):
        super().__init__()
        self.fused_dim = fused_dim
        self.risk_head = nn.Linear(fused_dim, 1)
        nn.init.xavier_uniform_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)
        self.weight_decay = kwargs.get('weight_decay', 1e-3); self.lr = kwargs.get('lr', 1e-3)
    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        return self.risk_head(fused_embedding).squeeze(-1)
    def fit(self, X_train: torch.Tensor, T_train: torch.Tensor, E_train: torch.Tensor, X_val: torch.Tensor, T_val: torch.Tensor, E_val: torch.Tensor, epochs: int = 200, patience: int = 20, verbose: bool = False) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_ci = 0.0; best_state = None; patience_counter = 0; history = {'train_loss': [], 'val_cindex': []}
        for epoch in range(epochs):
            self.train()
            risk = self.forward(X_train); loss = cox_partial_likelihood_loss(risk, T_train, E_train)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0); optimizer.step()
            self.eval()
            with torch.no_grad(): val_risk = self.forward(X_val).numpy()
            try: val_ci = concordance_index(T_val.numpy(), -val_risk, E_val.numpy())
            except Exception: val_ci = 0.5
            history['train_loss'].append(float(loss.item())); history['val_cindex'].append(float(val_ci))
            if val_ci > best_ci:
                best_ci = val_ci; best_state = {k: v.clone() for k, v in self.state_dict().items()}; patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience: break
        if best_state: self.load_state_dict(best_state)
        return {'best_val_cindex': best_ci, 'history': history}
    def predict_risk(self, X: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad(): return self.forward(X).numpy()
