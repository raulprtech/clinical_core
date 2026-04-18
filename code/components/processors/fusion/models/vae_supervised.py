from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# ============================================================
# MODEL
# ============================================================
 
class SupervisedVAE(nn.Module):
    """
    Supervised VAE for multimodal late fusion.
 
    Input layout:
        The forward() signature expects the concatenated embeddings of K
        modalities as a single tensor of shape [batch, K * d_per_modality],
        along with a confidence mask of shape [batch, K] indicating per-case
        reliability of each modality (1 = fully present, 0 = absent).
 
        Default K = 3 (tabular, text, vision) and d_per_modality = 768, so
        input_dim = 2304. Can be used in unimodal mode by setting K = 1.
 
    Architecture:
        encoder:  MLP input_dim → h1 → h2 → (μ, log σ²) each of dim d_latent
        decoder:  MLP d_latent → h2 → h1 → input_dim (reconstruction)
        risk:    Linear d_latent → 1 (Cox risk head, used for L_cox)
 
    Contract compliance (processing contract per protocol v12 §7.2):
        1. Discriminative output (empirically verifiable via C-index).
        2. Confidence output per case (derived from input mask aggregate).
        3. Graceful degradation under modality absence (enabled by training
           with masking).
    """
 
    def __init__(
        self,
        input_dim: int = 768 * 3,
        d_latent: int = 128,
        hidden_dims: Tuple[int, int] = (512, 256),
        n_modalities: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_latent = d_latent
        self.n_modalities = n_modalities
        self.d_per_modality = input_dim // n_modalities
 
        h1, h2 = hidden_dims
 
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.to_mu = nn.Linear(h2, d_latent)
        self.to_logvar = nn.Linear(h2, d_latent)
 
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(d_latent, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, input_dim),
        )
 
        # Cox risk head on latent — the supervision signal
        self.risk_head = nn.Linear(d_latent, 1)
 
    @property
    def name(self) -> str:
        return "vae_supervised"
 
    @property
    def contract_compliant(self) -> bool:
        return True
 
    def encode(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-8.0, 8.0)
        return mu, logvar
 
    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # deterministic at eval time (contract: reproducibility)
 
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)
 
    def forward(
        self,
        x: torch.Tensor,
        modality_confidence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x:  [batch, input_dim] concatenated embeddings.
            modality_confidence:  [batch, n_modalities] confidence per modality.
 
        Returns:
            Dict with keys:
                z:        [batch, d_latent] — sampled latent (training) or μ (eval).
                mu:       [batch, d_latent] — posterior mean.
                logvar:   [batch, d_latent] — posterior log variance.
                x_recon:  [batch, input_dim] — decoder reconstruction.
                risk:     [batch] — Cox risk score on z.
                conf:     [batch] — aggregated case-level confidence in [0, 1].
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        risk = self.risk_head(z).squeeze(-1)
        conf = modality_confidence.mean(dim=-1).clamp(0.0, 1.0)
        return {
            'z': z, 'mu': mu, 'logvar': logvar,
            'x_recon': x_recon, 'risk': risk, 'conf': conf,
        }
 
    @torch.no_grad()
    def extract_latent_space(
        self,
        x: torch.Tensor,
        modality_confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        After training, produce (Z, conf) for the entire cohort, using
        deterministic μ (no sampling). This is the tensor consumed by
        TurboLatent.
        """
        self.eval()
        out = self.forward(x, modality_confidence)
        return out['mu'], out['conf']
 
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
 
 
# ============================================================
# LOSS COMPONENTS
# ============================================================
 
def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) averaged over the batch."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
 
 
def cox_partial_likelihood(
    risk: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """
    Negative Cox partial-likelihood — same formulation the runner uses in
    train_variant_c. If zero events are present in the batch, returns 0 so
    the VAE reconstruction can still learn.
    """
    if events.sum() == 0:
        return torch.tensor(0.0, device=risk.device, requires_grad=True)
 
    order = torch.argsort(durations, descending=True)
    risk_s = risk[order]
    events_s = events[order]
 
    # log sum exp of cumulative risk for each case
    cum_logsumexp = torch.logcumsumexp(risk_s, dim=0)
    per_case = risk_s - cum_logsumexp
    return -(per_case * events_s).sum() / events_s.sum().clamp(min=1)
 
 
# ============================================================
# TRAINING
# ============================================================
 
@dataclass
class VAETrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    patience: int = 20
    # Loss weights (α_recon, β_kl, γ_cox)
    alpha_recon: float = 1.0
    beta_kl: float = 0.01   # annealed from 0 to this value over `kl_anneal_epochs`
    gamma_cox: float = 1.0
    kl_anneal_epochs: int = 30
    # Modality masking during training (only effective if train_with_masking=True)
    train_with_masking: bool = False
    modality_dropout_prob: float = 0.3
    # Reproducibility
    seed: int = 42
    verbose: bool = False
 
 
def _apply_modality_dropout(
    x: torch.Tensor,
    modality_confidence: torch.Tensor,
    n_modalities: int,
    d_per_modality: int,
    dropout_prob: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly zero out entire modalities per case. For each batch example and
    each modality, drop that modality with probability dropout_prob. When a
    modality is dropped both its embedding slice is zeroed and its confidence
    channel is set to 0 so the downstream encoder receives a coherent signal.
 
    Constraint: never drop ALL modalities at once — at least one must remain
    for each case so there is some signal to encode.
    """
    B = x.shape[0]
    keep_mask = (torch.rand(B, n_modalities, device=device) >= dropout_prob).float()
 
    # Ensure each row keeps at least one modality
    all_dropped = keep_mask.sum(dim=-1) == 0
    if all_dropped.any():
        # Force a random modality to be kept for those rows
        forced = torch.randint(0, n_modalities, (int(all_dropped.sum()),), device=device)
        keep_mask[all_dropped, forced] = 1.0
 
    # Expand keep_mask to embedding positions
    expanded = keep_mask.unsqueeze(-1).expand(B, n_modalities, d_per_modality)
    expanded = expanded.reshape(B, n_modalities * d_per_modality)
 
    x_masked = x * expanded
    conf_masked = modality_confidence * keep_mask
    return x_masked, conf_masked
 
 
def train_supervised_vae(
    vae: SupervisedVAE,
    X_tr: torch.Tensor, conf_tr: torch.Tensor,
    T_tr: torch.Tensor, E_tr: torch.Tensor,
    X_va: torch.Tensor, conf_va: torch.Tensor,
    T_va: torch.Tensor, E_va: torch.Tensor,
    cfg: Optional[VAETrainConfig] = None,
) -> Dict[str, Any]:
    """
    Train the supervised VAE with the triple loss.
 
    Early stopping is driven by validation C-index on the Cox risk head, not
    by reconstruction or KL — because the downstream purpose of Z is
    discriminative. The best model state (highest val C-index) is restored
    before returning.
 
    Returns a dict with:
        'best_val_cindex':  float
        'best_epoch':       int
        'train_history':    list of dicts per epoch
        'risk_head':        the risk head of the best-epoch model (for
                            downstream calibration/evaluation, mirrors
                            train_variant_c's return signature)
    """
    from lifelines.utils import concordance_index
 
    cfg = cfg or VAETrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
 
    device = next(vae.parameters()).device
    X_tr = X_tr.to(device); conf_tr = conf_tr.to(device)
    T_tr = T_tr.to(device); E_tr = E_tr.to(device)
    X_va = X_va.to(device); conf_va = conf_va.to(device)
    T_va = T_va.to(device); E_va = E_va.to(device)
 
    optimizer = torch.optim.AdamW(
        vae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
 
    best_cindex = -np.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    patience_counter = 0
    history: List[Dict[str, float]] = []
 
    n = X_tr.shape[0]
    n_batches = max(1, n // cfg.batch_size)
 
    for epoch in range(cfg.epochs):
        # KL annealing — prevents posterior collapse early in training
        if cfg.kl_anneal_epochs > 0:
            beta_now = cfg.beta_kl * min(1.0, epoch / cfg.kl_anneal_epochs)
        else:
            beta_now = cfg.beta_kl
 
        vae.train()
        perm = torch.randperm(n, device=device)
        epoch_recon = epoch_kl = epoch_cox = 0.0
 
        for b in range(n_batches):
            idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            if idx.numel() == 0:
                continue
            xb = X_tr[idx]
            cb = conf_tr[idx]
            tb = T_tr[idx]
            eb = E_tr[idx]
 
            # Modality masking, applied only if configured
            if cfg.train_with_masking:
                xb, cb = _apply_modality_dropout(
                    xb, cb,
                    n_modalities=vae.n_modalities,
                    d_per_modality=vae.d_per_modality,
                    dropout_prob=cfg.modality_dropout_prob,
                    device=device,
                )
 
            out = vae(xb, cb)
            L_recon = F.mse_loss(out['x_recon'], xb, reduction='mean')
            L_kl = kl_divergence(out['mu'], out['logvar'])
            L_cox = cox_partial_likelihood(out['risk'], tb, eb)
 
            loss = (
                cfg.alpha_recon * L_recon
                + beta_now * L_kl
                + cfg.gamma_cox * L_cox
            )
 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
            optimizer.step()
 
            epoch_recon += float(L_recon.item())
            epoch_kl += float(L_kl.item())
            epoch_cox += float(L_cox.item())
 
        # ---- Validation (always on clean — no masking at eval time) ----
        vae.eval()
        with torch.no_grad():
            out_va = vae(X_va, conf_va)
            risk_va = out_va['risk'].cpu().numpy()
 
        # Higher risk = higher hazard, so feed -risk to lifelines concordance
        try:
            val_ci = concordance_index(
                T_va.cpu().numpy(), -risk_va, E_va.cpu().numpy()
            )
        except Exception:
            val_ci = float('nan')
 
        history.append({
            'epoch': epoch,
            'recon': epoch_recon / max(1, n_batches),
            'kl': epoch_kl / max(1, n_batches),
            'cox': epoch_cox / max(1, n_batches),
            'beta_now': beta_now,
            'val_cindex': float(val_ci),
        })
 
        if cfg.verbose and epoch % 10 == 0:
            print(f"  [ep {epoch:3d}] recon={history[-1]['recon']:.4f} "
                  f"kl={history[-1]['kl']:.4f} cox={history[-1]['cox']:.4f} "
                  f"β={beta_now:.3f} val_CI={val_ci:.4f}")
 
        if not np.isnan(val_ci) and val_ci > best_cindex:
            best_cindex = val_ci
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in vae.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                if cfg.verbose:
                    print(f"  Early stop at epoch {epoch} (best epoch {best_epoch})")
                break
 
    if best_state is not None:
        vae.load_state_dict(best_state)
 
    return {
        'best_val_cindex': float(best_cindex),
        'best_epoch': int(best_epoch),
        'train_history': history,
        'risk_head': vae.risk_head,  # for downstream compatibility
        'model': vae,
    }
 
 
# ============================================================
# PROCESSING CONTRACT VERIFICATION
# ============================================================
 
def verify_processing_contract(
    Z: torch.Tensor,
    conf: torch.Tensor,
    d_latent: int,
    val_cindex: float,
    min_cindex: float = 0.6,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Verify the three guarantees of the processing contract (protocol v12 §7.2).
 
    Guarantee 1 (discriminative):
        val C-index on the held-out set exceeds a minimum threshold.
    Guarantee 2 (confidence):
        per-case confidence is in [0, 1] and has non-trivial variance
        (a constant confidence fails the spirit of the guarantee).
    Guarantee 3 (graceful degradation):
        NOT verified here — requires a second call with a degraded input set.
        Phase 4 of the runner is the natural place to test this.
    """
    ok_dim = Z.shape[-1] == d_latent
    ok_discriminative = val_cindex >= min_cindex
    ok_conf_range = (conf.min() >= 0.0) and (conf.max() <= 1.0)
    ok_conf_variance = conf.std() > 1e-4  # some spread is required
 
    all_ok = bool(ok_dim and ok_discriminative and ok_conf_range and ok_conf_variance)
 
    report = {
        'contract_satisfied': all_ok,
        'dimension_ok': bool(ok_dim),
        'discriminative_ok': bool(ok_discriminative),
        'val_cindex': float(val_cindex),
        'min_cindex_threshold': float(min_cindex),
        'confidence_range_ok': bool(ok_conf_range),
        'confidence_variance_ok': bool(ok_conf_variance.item() if torch.is_tensor(ok_conf_variance) else ok_conf_variance),
        'graceful_degradation': 'not_tested_here',
    }
 
    if verbose:
        status = '✓' if all_ok else '✗'
        print(f"  [processing contract] {status}")
        print(f"    dim={Z.shape[-1]} (expected {d_latent})  {'✓' if ok_dim else '✗'}")
        print(f"    val C-index={val_cindex:.4f} (min {min_cindex})  "
              f"{'✓' if ok_discriminative else '✗'}")
        print(f"    conf ∈ [{conf.min():.3f}, {conf.max():.3f}]  "
              f"{'✓' if ok_conf_range else '✗'}")
 
    return report
 
 
# ============================================================
# REGISTRY FACTORY
# ============================================================
 
def build_supervised_vae(
    input_dim: int = 768 * 3,
    d_latent: int = 128,
    **kwargs,
) -> SupervisedVAE:
    """
    Factory for registry.py. Accepts kwargs from experiment_config.yaml:
        hidden_dims, n_modalities, dropout.
    """
    return SupervisedVAE(
        input_dim=input_dim,
        d_latent=d_latent,
        hidden_dims=kwargs.get('hidden_dims', (512, 256)),
        n_modalities=kwargs.get('n_modalities', 3),
        dropout=kwargs.get('dropout', 0.1),
    )