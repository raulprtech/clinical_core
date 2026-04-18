"""
PROGNOSIS-PROC: Weibull Parametric Head (v2 — fixed initialization)
====================================================================

Paramteric survival head that predicts a full survival curve S(t|Z) via
Weibull distribution parameters (k, λ), not just a relative risk score.

Motivation over linear Cox:
    Cox gives relative ordering of risk but cannot produce calibrated
    survival probabilities at specific time horizons (1yr, 3yr, 5yr).
    Weibull parametrizes the hazard function directly, enabling:
      1. S(t|Z) for any t without non-parametric baseline estimation.
      2. Time-specific survival probabilities for calibration plots.
      3. Decision Curve Analysis at arbitrary thresholds.

Parametrization:
    For each patient z, the head outputs two parameters:
        k(z)  = softplus(w_k · z + b_k) + 1e-3   (shape parameter, > 0)
        λ(z)  = softplus(w_λ · z + b_λ) + 1e-3   (scale parameter, > 0)

    Survival function: S(t|z) = exp(-(t/λ(z))^k(z))
    Risk score (for C-index): -log(λ(z)).

Initialization fix (v2):
    λ must start close to the cohort's median survival time (~1000 days
    for TCGA-KIRC) so that (t/λ)^k stays in a tractable range during the
    first epochs. Since softplus(x) ≈ x for large positive x, the scale
    bias is initialized DIRECTLY to init_scale (e.g. 1000.0), not to
    log(init_scale). A previous version used log(1000)≈7 as bias, which
    produced λ≈7 and collapsed all survival curves to zero — documented
    as a lesson learned for the quarterly report.

Training loss:
    Negative log-likelihood of Weibull with right censoring.
        uncensored: log f(t) = log(k/λ) + (k-1)·log(t/λ) - (t/λ)^k
        censored:   log S(t) = -(t/λ)^k
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
    """Negative log-likelihood of Weibull with right censoring."""
    eps = 1e-7
    durations = durations.clamp(min=eps)
    k = k.clamp(min=eps)
    lam = lam.clamp(min=eps)

    t_over_lam = durations / lam
    log_t_over_lam = torch.log(t_over_lam + eps)

    log_hazard = torch.log(k / lam + eps) + (k - 1.0) * log_t_over_lam
    cum_hazard = t_over_lam.pow(k)

    log_lik = events * log_hazard - cum_hazard
    return -log_lik.mean()


class PrognosisProc_WeibullHead(nn.Module):
    """
    Weibull parametric survival head consuming a latent Z.

    Args:
        fused_dim:  input dimensionality (= d_latent of FUSION-PROC).
        init_scale: expected median survival time, in the same units as
                    durations (default 1000.0 days ≈ TCGA-KIRC median).
        init_shape: expected Weibull shape at initialization (default 1.0).
    """

    name = "prognosis_weibull_head"

    def __init__(
        self,
        fused_dim: int,
        init_scale: float = 1000.0,
        init_shape: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.fused_dim = fused_dim
        self.shape_head = nn.Linear(fused_dim, 1)
        self.scale_head = nn.Linear(fused_dim, 1)

        nn.init.xavier_uniform_(self.shape_head.weight)
        nn.init.xavier_uniform_(self.scale_head.weight)

        # Initialize biases directly to the desired raw values.
        # softplus(x) ≈ x for x >> 0, so bias=1000 gives λ≈1000 at epoch 0.
        nn.init.constant_(self.scale_head.bias, float(init_scale))
        nn.init.constant_(self.shape_head.bias, float(init_shape))

        self.weight_decay = kwargs.get('weight_decay', 1e-3)
        self.lr = kwargs.get('lr', 1e-3)

    def forward(self, fused_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        k_raw = self.shape_head(fused_embedding).squeeze(-1)
        lam_raw = self.scale_head(fused_embedding).squeeze(-1)
        k = torch.nn.functional.softplus(k_raw) + 1e-3
        lam = torch.nn.functional.softplus(lam_raw) + 1e-3
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
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        best_ci = 0.0
        best_state = None
        patience_counter = 0
        history = {
            'train_loss': [], 'val_cindex': [],
            'k_mean': [], 'k_std': [],
            'lam_mean': [], 'lam_std': [],
        }

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
                history['k_mean'].append(float(val_out['k'].mean()))
                history['k_std'].append(float(val_out['k'].std()))
                history['lam_mean'].append(float(val_out['lam'].mean()))
                history['lam_std'].append(float(val_out['lam'].std()))

            try:
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
                      f"val_CI={val_ci:.4f}  "
                      f"λ̄={history['lam_mean'][-1]:.1f}  "
                      f"k̄={history['k_mean'][-1]:.2f}")

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
            times: [T] numpy array of time points (same units as durations)

        Returns:
            [batch, T] numpy array of survival probabilities in [0, 1].
        """
        self.eval()
        out = self(X)
        k = out['k'].cpu().numpy()
        lam = out['lam'].cpu().numpy()
        t = times[np.newaxis, :]
        k_b = k[:, np.newaxis]
        lam_b = lam[:, np.newaxis]
        # Clip exponent to prevent overflow when λ is small or t is large
        exponent = np.clip(np.power(t / lam_b, k_b), 0, 700)
        return np.exp(-exponent)


def build_weibull_head(fused_dim: int, **kwargs) -> PrognosisProc_WeibullHead:
    """Factory for registry.py."""
    return PrognosisProc_WeibullHead(fused_dim=fused_dim, **kwargs)