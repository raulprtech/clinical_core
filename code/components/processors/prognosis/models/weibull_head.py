"""
PROGNOSIS-PROC: Weibull Parametric Head
========================================

Paramteric survival head that predicts a full survival curve S(t|Z) via
Weibull distribution parameters (k, λ), not just a relative risk score.

Motivation over linear Cox:
    Cox gives relative ordering of risk but cannot produce calibrated
    survival probabilities at specific time horizons (1yr, 3yr, 5yr).
    Weibull parametrizes the hazard function directly, enabling:
      1. S(t|Z) for any t without non-parametric baseline estimation.
      2. Time-specific survival probabilities for calibration plots.
      3. Decision Curve Analysis at arbitrary thresholds.

Protocol v12 §4.2 OE3: "PROGNOSIS-PROC implemented as Weibull head,
decoupable from the fusion processor". This module is that implementation.

Parametrization:
    For each patient z, the head outputs two parameters:
        k(z)  = softplus(w_k · z + b_k) + 1e-3   (shape parameter, > 0)
        λ(z)  = softplus(w_λ · z + b_λ) + 1e-3   (scale parameter, > 0)
    
    Survival function: S(t|z) = exp(-(t/λ(z))^k(z))
    Risk score (for C-index): -log(λ(z)) gives a scalar ordering that
        correlates with hazard; used only for concordance computation.

Training loss:
    Negative log-likelihood of the Weibull distribution with right censoring.
    For uncensored patients: log f(t) = log(k/λ) + (k-1)·log(t/λ) - (t/λ)^k
    For censored patients:   log S(t) = -(t/λ)^k
    
    NLL = -Σ_uncensored log f(t_i | z_i) - Σ_censored log S(t_i | z_i)

This implementation follows Chapfuwa et al. (2019) on calibrated Weibull
networks for survival and adapts it to operate on top of a frozen latent
space produced by FUSION-PROC.
"""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index


def _weibull_nll(
    k: torch.Tensor,
    lam: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """
    Negative log-likelihood of Weibull with right censoring.

    Args:
        k:         shape parameter, [batch], > 0
        lam:       scale parameter, [batch], > 0
        durations: survival/censoring times, [batch], > 0
        events:    1 if event observed, 0 if censored, [batch]
    """
    # Numerical safeguards
    eps = 1e-7
    durations = durations.clamp(min=eps)
    k = k.clamp(min=eps)
    lam = lam.clamp(min=eps)

    t_over_lam = durations / lam
    log_t_over_lam = torch.log(t_over_lam + eps)

    # Log-hazard for uncensored: log(k/λ) + (k-1)·log(t/λ)
    log_hazard = torch.log(k / lam + eps) + (k - 1.0) * log_t_over_lam
    # Cumulative hazard: (t/λ)^k, log-survival = -cum hazard
    cum_hazard = t_over_lam.pow(k)

    # NLL: uncensored contribute log_hazard - cum_hazard;
    #      censored contribute -cum_hazard only
    log_lik = events * log_hazard - cum_hazard
    return -log_lik.mean()


class PrognosisProc_WeibullHead(nn.Module):
    """
    Weibull parametric survival head that consumes a latent Z and outputs
    survival curve parameters (k, λ) per patient.
    """

    name = "prognosis_weibull_head"

    def __init__(self, fused_dim: int, **kwargs):
        super().__init__()
        self.fused_dim = fused_dim
        # Two separate linear heads — one for each Weibull parameter
        self.shape_head = nn.Linear(fused_dim, 1)
        self.scale_head = nn.Linear(fused_dim, 1)

        nn.init.xavier_uniform_(self.shape_head.weight)
        nn.init.zeros_(self.shape_head.bias)
        nn.init.xavier_uniform_(self.scale_head.weight)
        # Initialize scale bias so λ starts around a reasonable prior
        # (median survival time in TCGA-KIRC is ~1000 days → log(1000) ≈ 7)
        nn.init.constant_(self.scale_head.bias, 7.0)

        self.weight_decay = kwargs.get('weight_decay', 1e-3)
        self.lr = kwargs.get('lr', 1e-3)

    def forward(self, fused_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns dict with k, lam, and risk (scalar for C-index).
        All tensors have shape [batch].
        """
        k_raw = self.shape_head(fused_embedding).squeeze(-1)
        lam_raw = self.scale_head(fused_embedding).squeeze(-1)
        # softplus keeps params positive; +eps guards against exactly zero
        k = torch.nn.functional.softplus(k_raw) + 1e-3
        lam = torch.nn.functional.softplus(lam_raw) + 1e-3
        # Use -log(λ) as scalar risk ordering. Shorter scale → higher risk.
        risk = -torch.log(lam + 1e-7)
        return {'k': k, 'lam': lam, 'risk': risk}

    def fit(
        self,
        X_train: torch.Tensor, T_train: torch.Tensor, E_train: torch.Tensor,
        X_val: torch.Tensor, T_val: torch.Tensor, E_val: torch.Tensor,
        epochs: int = 200,
        patience: int = 20,
        verbose: bool = False,
    ) -> dict:
        """
        Train on a frozen latent Z (X_train/X_val are Z from FUSION-PROC).

        Compatible with the fit() signature used by main.py's PROGNOSIS-PROC
        invocation, so it drops into the existing pipeline without runner
        changes.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        best_ci = 0.0
        best_state = None
        patience_counter = 0
        history = {'train_loss': [], 'val_cindex': []}

        for epoch in range(epochs):
            self.train()
            out = self(X_train)
            loss = _weibull_nll(out['k'], out['lam'], T_train, E_train)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_out = self(X_val)
                val_risk = val_out['risk'].cpu().numpy()
            try:
                # For Weibull, higher -log(λ) means shorter expected survival,
                # so risk itself (not -risk) is the hazard-like ordering
                val_ci = concordance_index(
                    T_val.cpu().numpy(), -val_risk, E_val.cpu().numpy(),
                )
            except Exception:
                val_ci = 0.5

            history['train_loss'].append(float(loss.item()))
            history['val_cindex'].append(float(val_ci))

            if val_ci > best_ci:
                best_ci = val_ci
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            if verbose and epoch % 20 == 0:
                print(f"  [ep {epoch:3d}] NLL={loss.item():.4f}  "
                      f"val_CI={val_ci:.4f}")

        if best_state:
            self.load_state_dict(best_state)
        return {'best_val_cindex': best_ci, 'history': history}

    def predict_risk(self, X: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self(X)['risk'].cpu().numpy()

    @torch.no_grad()
    def predict_survival(
        self, X: torch.Tensor, times: np.ndarray,
    ) -> np.ndarray:
        """
        Predict S(t|z) for each patient at each requested time.
        
        Args:
            X:     [batch, fused_dim] latent space
            times: [T] numpy array of time points in the same units as
                   durations during training (typically days)
        
        Returns:
            [batch, T] numpy array of survival probabilities.
        """
        self.eval()
        out = self(X)
        k = out['k'].cpu().numpy()
        lam = out['lam'].cpu().numpy()
        # Broadcast: times shape [T] vs params shape [batch]
        # S(t|z) = exp(-(t/λ)^k)
        t = times[np.newaxis, :]              # [1, T]
        k_b = k[:, np.newaxis]                # [batch, 1]
        lam_b = lam[:, np.newaxis]            # [batch, 1]
        return np.exp(-np.power(t / lam_b, k_b))


def build_weibull_head(fused_dim: int, **kwargs) -> PrognosisProc_WeibullHead:
    """Factory for registry.py."""
    return PrognosisProc_WeibullHead(fused_dim=fused_dim, **kwargs)