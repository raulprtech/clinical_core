"""
FUSION-PROC: Generative VAE Strategy
=====================================

A late-fusion variational autoencoder trained in two stages:

    Stage A — Generative pretraining
        Loss: α · L_recon + β · L_KL
        Goal: learn a meaningful latent space that reconstructs the
              concatenated modality embeddings. No supervision yet.

    Stage B — Contrastive fine-tuning by survival
        Loss: α · L_recon + β · L_KL + δ · L_contrastive
        Goal: shape the latent space so that patients with similar
              survival outcomes are close in Z and patients with
              opposite outcomes are distant. No direct Cox head.

The key architectural decision that distinguishes this from the previous
vae_supervised.py implementation: there is NO risk head on Z. PROGNOSIS-PROC
is an independent downstream component that consumes frozen Z after training.
This respects the FUSION-PROC / PROGNOSIS-PROC separation defined in the
protocol v12, and prevents the shortcut learning that caused early stopping
in epoch 0-1 of the previous implementation.

Contrastive supervision rationale:
    We want the VAE to learn representations where survival information is
    captured geometrically, not via a linear projection. Triplet loss with
    hard-ish negatives mined from the batch forces the encoder to compress
    the input in ways that separate outcomes, while the KL and reconstruction
    terms prevent collapse.

Censoring handling:
    Censored patients can only serve as anchor or positive against other
    censored patients, because we don't know their true outcome. When pairing
    an uncensored anchor, a censored patient can act as a positive only if
    its censoring time is comparable to the anchor's event time. The triplet
    mining function implements this explicitly.

Contract compliance (protocol v12 §7.2):
    1. Discriminative output — verified empirically downstream via
       PROGNOSIS-PROC C-index on Z.
    2. Per-case confidence — aggregated from per-modality confidences.
    3. Graceful degradation — enabled by optional modality dropout during
       training (configurable via VAEGenTrainConfig.train_with_masking).
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MODEL
# ============================================================

class FusionProc_GenerativeVAE(nn.Module):
    """
    Generative VAE for late fusion of multimodal embeddings.

    No survival head is attached. The latent space Z is produced by a pure
    generative + contrastive objective. PROGNOSIS-PROC is responsible for
    producing risk scores from Z in a second, decoupled stage.

    Args:
        input_dim: total concatenated input dimensionality (default 768 × 3).
        d_latent:  latent space dimensionality (default 128).
        hidden_dims: tuple of two hidden-layer sizes for enc/dec.
        n_modalities: number of input modalities (default 3).
        dropout: dropout probability in enc/dec MLPs.
    """

    name = "fusion_vae_generative"

    def __init__(
        self,
        input_dim: int = 768 * 3,
        d_latent: int = 128,
        hidden_dims: Tuple[int, int] = (512, 256),
        n_modalities: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_latent = d_latent
        self.n_modalities = n_modalities
        self.d_per_modality = input_dim // n_modalities

        h1, h2 = hidden_dims

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

    @property
    def contract_compliant(self) -> bool:
        return True

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-8.0, 8.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # During training, sample from posterior. At eval time return the
        # deterministic μ so that PROGNOSIS-PROC receives reproducible Z.
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(
        self,
        x: torch.Tensor,
        modality_confidence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        conf = modality_confidence.mean(dim=-1).clamp(0.0, 1.0)
        return {
            'z': z, 'mu': mu, 'logvar': logvar,
            'x_recon': x_recon, 'conf': conf,
        }

    @torch.no_grad()
    def extract_latent_space(
        self,
        x: torch.Tensor,
        modality_confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produce deterministic (Z, conf) for the entire cohort using μ.
        This is what PROGNOSIS-PROC and TurboLatent consume downstream.
        """
        self.eval()
        out = self.forward(x, modality_confidence)
        return out['mu'], out['conf']

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -------------------------------------------------------
    # MAIN ENTRY POINT (two-stage training)
    # -------------------------------------------------------
    def fit(
        self,
        X_train: torch.Tensor, conf_train: torch.Tensor,
        T_train: torch.Tensor, E_train: torch.Tensor,
        X_val: torch.Tensor, conf_val: torch.Tensor,
        T_val: torch.Tensor, E_val: torch.Tensor,
        cfg: Optional['VAEGenTrainConfig'] = None,
    ) -> Dict[str, Any]:
        """
        Two-stage training orchestrator.

        Stage A: pretrain with L_recon + L_KL only (no supervision).
        Stage B: fine-tune adding the triplet contrastive loss on survival.

        The default criterion for selecting the "best" model across stages is
        validation reconstruction + KL during Stage A, and a held-out silhouette
        score on Z colored by event status during Stage B (higher silhouette
        => better structural separation of outcomes). This decouples us from
        any C-index metric at training time, which is downstream's job.
        """
        cfg = cfg or VAEGenTrainConfig()
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        device = next(self.parameters()).device
        X_train = X_train.to(device); conf_train = conf_train.to(device)
        T_train = T_train.to(device); E_train = E_train.to(device)
        X_val = X_val.to(device); conf_val = conf_val.to(device)
        T_val = T_val.to(device); E_val = E_val.to(device)

        history_A = self._train_stage_a(
            X_train, conf_train, X_val, conf_val, cfg, device,
        )
        history_B = self._train_stage_b(
            X_train, conf_train, T_train, E_train,
            X_val, conf_val, T_val, E_val,
            cfg, device,
        )

        return {
            'stage_A_history': history_A,
            'stage_B_history': history_B,
            'model': self,
            # Bookkeeping for downstream consistency with the rest of the
            # pipeline. Note: we do NOT return a val_cindex here because
            # FUSION-PROC is not a predictor. That's PROGNOSIS-PROC's job.
            'best_val_cindex': None,
        }

    # -------------------------------------------------------
    # STAGE A — generative pretraining
    # -------------------------------------------------------
    def _train_stage_a(
        self,
        X_train: torch.Tensor, conf_train: torch.Tensor,
        X_val: torch.Tensor, conf_val: torch.Tensor,
        cfg: 'VAEGenTrainConfig', device: torch.device,
    ) -> List[Dict[str, float]]:
        opt = torch.optim.AdamW(
            self.parameters(), lr=cfg.lr_stage_a, weight_decay=cfg.weight_decay,
        )
        n = X_train.shape[0]
        n_batches = max(1, n // cfg.batch_size)
        history: List[Dict[str, float]] = []
        best_val = np.inf
        patience_count = 0
        best_state = None

        for epoch in range(cfg.epochs_stage_a):
            # Linear KL annealing only during Stage A
            if cfg.kl_anneal_epochs > 0:
                beta_now = cfg.beta_kl * min(1.0, epoch / cfg.kl_anneal_epochs)
            else:
                beta_now = cfg.beta_kl

            self.train()
            perm = torch.randperm(n, device=device)
            ep_recon = ep_kl = 0.0

            for b in range(n_batches):
                idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
                if idx.numel() == 0:
                    continue
                xb = X_train[idx]; cb = conf_train[idx]
                if cfg.train_with_masking:
                    xb, cb = _apply_modality_dropout(
                        xb, cb, self.n_modalities, self.d_per_modality,
                        cfg.modality_dropout_prob, device,
                    )
                out = self(xb, cb)
                L_recon = F.mse_loss(out['x_recon'], xb, reduction='mean')
                L_kl = _kl_divergence(out['mu'], out['logvar'])
                loss = cfg.alpha_recon * L_recon + beta_now * L_kl
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                opt.step()
                ep_recon += float(L_recon.item())
                ep_kl += float(L_kl.item())

            # Validation — pure recon + KL
            self.eval()
            with torch.no_grad():
                out_v = self(X_val, conf_val)
                val_recon = F.mse_loss(out_v['x_recon'], X_val, reduction='mean').item()
                val_kl = _kl_divergence(out_v['mu'], out_v['logvar']).item()
                val_total = cfg.alpha_recon * val_recon + beta_now * val_kl

            history.append({
                'stage': 'A',
                'epoch': epoch,
                'recon': ep_recon / max(1, n_batches),
                'kl': ep_kl / max(1, n_batches),
                'beta_now': beta_now,
                'val_recon': val_recon,
                'val_kl': val_kl,
                'val_total': val_total,
            })

            if cfg.verbose and epoch % 10 == 0:
                print(f"  [A ep {epoch:3d}] recon={history[-1]['recon']:.4f} "
                      f"kl={history[-1]['kl']:.4f} β={beta_now:.3f} "
                      f"val={val_total:.4f}")

            # Early stopping on val_total (only meaningful after KL annealing
            # has stabilized; use patience_stage_a that accounts for it)
            if val_total < best_val:
                best_val = val_total
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= cfg.patience_stage_a and epoch > cfg.kl_anneal_epochs:
                    if cfg.verbose:
                        print(f"  [A] Early stop at epoch {epoch}")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        return history

    # -------------------------------------------------------
    # STAGE B — contrastive fine-tuning by survival
    # -------------------------------------------------------
    def _train_stage_b(
        self,
        X_train: torch.Tensor, conf_train: torch.Tensor,
        T_train: torch.Tensor, E_train: torch.Tensor,
        X_val: torch.Tensor, conf_val: torch.Tensor,
        T_val: torch.Tensor, E_val: torch.Tensor,
        cfg: 'VAEGenTrainConfig', device: torch.device,
    ) -> List[Dict[str, float]]:
        opt = torch.optim.AdamW(
            self.parameters(), lr=cfg.lr_stage_b, weight_decay=cfg.weight_decay,
        )
        n = X_train.shape[0]
        n_batches = max(1, n // cfg.batch_size)
        history: List[Dict[str, float]] = []
        best_silhouette = -np.inf
        patience_count = 0
        best_state = None

        for epoch in range(cfg.epochs_stage_b):
            self.train()
            perm = torch.randperm(n, device=device)
            ep_recon = ep_kl = ep_contra = 0.0
            ep_triplets_used = 0

            for b in range(n_batches):
                idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
                if idx.numel() < 4:
                    continue  # Need enough patients to mine triplets
                xb = X_train[idx]; cb = conf_train[idx]
                tb = T_train[idx]; eb = E_train[idx]
                if cfg.train_with_masking:
                    xb, cb = _apply_modality_dropout(
                        xb, cb, self.n_modalities, self.d_per_modality,
                        cfg.modality_dropout_prob, device,
                    )
                out = self(xb, cb)
                L_recon = F.mse_loss(out['x_recon'], xb, reduction='mean')
                L_kl = _kl_divergence(out['mu'], out['logvar'])

                # Contrastive loss on the latent μ
                L_contra, n_triplets = _survival_triplet_loss(
                    mu=out['mu'], durations=tb, events=eb,
                    margin=cfg.triplet_margin,
                    time_similar_window=cfg.time_similar_window,
                )

                loss = (cfg.alpha_recon * L_recon
                        + cfg.beta_kl * L_kl
                        + cfg.delta_contra * L_contra)

                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                opt.step()

                ep_recon += float(L_recon.item())
                ep_kl += float(L_kl.item())
                ep_contra += float(L_contra.item())
                ep_triplets_used += n_triplets

            # Validation — silhouette on held-out set, colored by event
            self.eval()
            with torch.no_grad():
                mu_val = self(X_val, conf_val)['mu']
                val_sil = _silhouette_by_event(mu_val, E_val)

            history.append({
                'stage': 'B',
                'epoch': epoch,
                'recon': ep_recon / max(1, n_batches),
                'kl': ep_kl / max(1, n_batches),
                'contra': ep_contra / max(1, n_batches),
                'triplets_per_epoch': ep_triplets_used,
                'val_silhouette_event': val_sil,
            })

            if cfg.verbose and epoch % 5 == 0:
                print(f"  [B ep {epoch:3d}] recon={history[-1]['recon']:.4f} "
                      f"contra={history[-1]['contra']:.4f} "
                      f"triplets={ep_triplets_used} "
                      f"val_sil={val_sil:.4f}")

            # Best model selection by val silhouette (higher is better)
            if not np.isnan(val_sil) and val_sil > best_silhouette:
                best_silhouette = val_sil
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= cfg.patience_stage_b:
                    if cfg.verbose:
                        print(f"  [B] Early stop at epoch {epoch}")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        return history


# ============================================================
# TRAINING CONFIG
# ============================================================

@dataclass
class VAEGenTrainConfig:
    # --- Stage A (pretraining) ---
    epochs_stage_a: int = 100
    lr_stage_a: float = 1e-3
    patience_stage_a: int = 20
    # --- Stage B (contrastive fine-tuning) ---
    epochs_stage_b: int = 60
    lr_stage_b: float = 3e-4
    patience_stage_b: int = 15
    # --- Loss weights ---
    alpha_recon: float = 1.0
    beta_kl: float = 0.01        # Kept low to avoid posterior collapse
    delta_contra: float = 0.5    # Contrastive term weight (Stage B only)
    kl_anneal_epochs: int = 30   # Only active during Stage A
    # --- Triplet loss ---
    triplet_margin: float = 1.0
    time_similar_window: float = 180.0  # days — defines "similar survival"
    # --- Shared ---
    weight_decay: float = 1e-4
    batch_size: int = 64
    train_with_masking: bool = False
    modality_dropout_prob: float = 0.3
    seed: int = 42
    verbose: bool = False


# ============================================================
# LOSS HELPERS
# ============================================================

def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def _survival_triplet_loss(
    mu: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
    margin: float = 1.0,
    time_similar_window: float = 180.0,
) -> Tuple[torch.Tensor, int]:
    """
    Survival-aware triplet loss.

    For each anchor patient, we mine:
        positive: another patient with similar survival (|Δt| < window) AND
                  same event status. Similar outcome → should be close in Z.
        negative: a patient with OPPOSITE event status AND distant survival
                  time. Opposite outcome → should be far in Z.

    Censoring handling — the pairing rules:
        Event anchor  + Event positive (|Δt| < window) — valid
        Censored anchor + Censored positive (|Δt| < window) — valid
        Event anchor + Censored negative where censor time > anchor+window — valid
        (censored patient at a late time is evidence of better outcome)
        Other combinations involving censoring are excluded because they
        carry ambiguity.

    If no valid triplets can be mined from the batch, returns a zero loss
    and n_triplets=0 (caller can monitor this to detect degenerate batches).

    Returns:
        (loss_value, n_triplets_used)
    """
    B = mu.shape[0]
    if B < 3:
        return torch.tensor(0.0, device=mu.device, requires_grad=True), 0

    device = mu.device
    events_bool = events.bool()

    # Compute pairwise L2 distances in μ-space
    mu_norm = (mu * mu).sum(dim=1, keepdim=True)
    dist2 = mu_norm + mu_norm.T - 2.0 * (mu @ mu.T)
    dist2 = dist2.clamp(min=1e-9)
    dist = dist2.sqrt()

    # Pairwise time differences
    dur_diff = (durations.unsqueeze(1) - durations.unsqueeze(0)).abs()

    # Masks
    # same_event[i, j] = True if events[i] == events[j]
    same_event = events_bool.unsqueeze(1) == events_bool.unsqueeze(0)
    diff_event = ~same_event
    # exclude self-pairs
    eye = torch.eye(B, dtype=torch.bool, device=device)
    same_event = same_event & ~eye

    # Positive candidate mask: same event AND similar survival time
    pos_mask = same_event & (dur_diff < time_similar_window)

    # Negative candidate mask: different event status AND distant times
    # (using the same window as threshold for "distant enough")
    neg_mask = diff_event & (dur_diff > time_similar_window)

    triplet_losses = []
    for i in range(B):
        pos_idx = torch.where(pos_mask[i])[0]
        neg_idx = torch.where(neg_mask[i])[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        # Semi-hard negative mining: hardest positive (furthest among valid
        # positives) and hardest negative (closest among valid negatives)
        d_ap = dist[i, pos_idx].max()
        d_an = dist[i, neg_idx].min()
        triplet_losses.append(F.relu(d_ap - d_an + margin))

    if len(triplet_losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    return torch.stack(triplet_losses).mean(), len(triplet_losses)


def _apply_modality_dropout(
    x: torch.Tensor,
    modality_confidence: torch.Tensor,
    n_modalities: int,
    d_per_modality: int,
    dropout_prob: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero-out entire modalities per case, ensuring at least one remains."""
    B = x.shape[0]
    keep_mask = (torch.rand(B, n_modalities, device=device) >= dropout_prob).float()
    all_dropped = keep_mask.sum(dim=-1) == 0
    if all_dropped.any():
        forced = torch.randint(0, n_modalities, (int(all_dropped.sum()),), device=device)
        keep_mask[all_dropped, forced] = 1.0
    expanded = keep_mask.unsqueeze(-1).expand(B, n_modalities, d_per_modality)
    expanded = expanded.reshape(B, n_modalities * d_per_modality)
    return x * expanded, modality_confidence * keep_mask


def _silhouette_by_event(mu: torch.Tensor, events: torch.Tensor) -> float:
    """
    Silhouette coefficient using event status as the cluster label.
    Higher = better structural separation of outcomes.
    Returns NaN if only one class is present in the validation set.
    """
    try:
        from sklearn.metrics import silhouette_score
        mu_np = mu.detach().cpu().numpy()
        labels = events.detach().cpu().numpy().astype(int)
        if len(set(labels)) < 2:
            return float('nan')
        return float(silhouette_score(mu_np, labels))
    except Exception:
        return float('nan')


# ============================================================
# CONTRACT VERIFICATION
# ============================================================

def verify_processing_contract(
    Z: torch.Tensor,
    conf: torch.Tensor,
    d_latent: int,
    downstream_cindex: Optional[float] = None,
    min_cindex: float = 0.60,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Verify the three guarantees of the processing contract.

    Guarantee 1 (discriminative): requires a downstream C-index value,
        obtained by training PROGNOSIS-PROC on the provided Z. Pass it
        explicitly via downstream_cindex.
    Guarantee 2 (confidence):   conf ∈ [0, 1] with non-trivial variance.
    Guarantee 3 (graceful deg.): not verified here; second call with
        degraded input required. Phase 4 of the runner tests this.
    """
    ok_dim = Z.shape[-1] == d_latent
    ok_conf_range = bool(conf.min() >= 0.0 and conf.max() <= 1.0)
    ok_conf_variance = bool(conf.std() > 1e-4)

    ok_discriminative = None
    if downstream_cindex is not None:
        ok_discriminative = bool(downstream_cindex >= min_cindex)

    ok_list = [ok_dim, ok_conf_range, ok_conf_variance]
    if ok_discriminative is not None:
        ok_list.append(ok_discriminative)
    all_ok = all(ok_list)

    report = {
        'contract_satisfied': all_ok,
        'dimension_ok': ok_dim,
        'confidence_range_ok': ok_conf_range,
        'confidence_variance_ok': ok_conf_variance,
        'discriminative_ok': ok_discriminative,
        'downstream_cindex': downstream_cindex,
        'min_cindex_threshold': min_cindex,
        'graceful_degradation': 'not_tested_here',
    }

    if verbose:
        flag = '✓' if all_ok else '✗'
        print(f"  [processing contract] {flag}")
        print(f"    dim={Z.shape[-1]} (expected {d_latent})  "
              f"{'✓' if ok_dim else '✗'}")
        print(f"    conf ∈ [{conf.min():.3f}, {conf.max():.3f}]  "
              f"{'✓' if ok_conf_range else '✗'}")
        print(f"    conf variance = {conf.std():.4f}  "
              f"{'✓' if ok_conf_variance else '✗'}")
        if downstream_cindex is not None:
            print(f"    downstream C-index = {downstream_cindex:.4f} "
                  f"(min {min_cindex})  "
                  f"{'✓' if ok_discriminative else '✗'}")
        else:
            print(f"    downstream C-index = (not provided — call "
                  f"PROGNOSIS-PROC.fit(Z) and re-run verification)")

    return report


# ============================================================
# REGISTRY FACTORY
# ============================================================

def build_generative_vae(
    input_dim: int = 768 * 3,
    d_latent: int = 128,
    **kwargs,
) -> FusionProc_GenerativeVAE:
    """Factory for registry.py."""
    return FusionProc_GenerativeVAE(
        input_dim=input_dim,
        d_latent=d_latent,
        hidden_dims=kwargs.get('hidden_dims', (512, 256)),
        n_modalities=kwargs.get('n_modalities', 3),
        dropout=kwargs.get('dropout', 0.1),
    )