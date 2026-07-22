import torch
import torch.nn as nn
import time
import numpy as np
from typing import Tuple, Dict

def verify_ingestion_contract(embedding: torch.Tensor, confidence: torch.Tensor, expected_dim: int = 768, verbose: bool = True) -> dict:
    results = {"dimensions_ok": False, "range_ok": False, "confidence_ok": False}
    if embedding.shape[-1] == expected_dim: results["dimensions_ok"] = True
    if torch.all(confidence >= 0.0) and torch.all(confidence <= 1.0): results["confidence_ok"] = True
    norms = torch.norm(embedding, p=2, dim=-1)
    if torch.allclose(norms, torch.ones_like(norms), atol=1e-2): results["range_ok"] = True
    results["contract_satisfied"] = all(results.values())
    if verbose: print(f"Contract Verification: {results}")
    return results

def cox_partial_likelihood_loss(risk_scores: torch.Tensor, survival_times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    sorted_idx = torch.argsort(survival_times, descending=True)
    risk_sorted = risk_scores[sorted_idx]
    events_sorted = events[sorted_idx]
    log_cumsum = torch.logcumsumexp(risk_sorted, dim=0)
    uncensored = risk_sorted - log_cumsum
    event_mask = events_sorted.bool()
    if event_mask.sum() == 0: return torch.tensor(0.0, requires_grad=True)
    return -uncensored[event_mask].mean()

def train_variant_c(model, X_train, M_train, T_train, E_train, X_val, M_val, T_val, E_val, epochs=200, lr=0.001, patience=20, weight_decay=0.0, verbose=False) -> dict:
    """
    Train an encoder + risk head with Cox partial-likelihood loss.

    Returns
    -------
    dict with keys:
        best_val_cindex    : final C-index on validation set (uses best-state weights)
        risk_head          : trained nn.Linear(output_dim, 1) for downstream prediction
        train_loss_history : list[float], per-epoch training loss   (NEW)
        val_loss_history   : list[float], per-epoch validation loss (NEW)
        best_epoch         : int, epoch with lowest val_loss        (NEW)
        n_epochs_run       : int, total epochs actually executed    (NEW)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_train, M_train, T_train, E_train = X_train.to(device), M_train.to(device), T_train.to(device), E_train.to(device)
    X_val, M_val, T_val, E_val = X_val.to(device), M_val.to(device), T_val.to(device), E_val.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    risk_head = nn.Linear(model.output_dim, 1).to(device)
    optimizer.add_param_group({'params': risk_head.parameters()})
    best_loss = float('inf'); best_state = None; patience_cnt = 0
    # NEW: per-epoch loss tracking for overfitting / underfitting diagnostics
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train(); risk_head.train()
        optimizer.zero_grad()
        emb, _ = model(X_train, M_train); risk = risk_head(emb).squeeze(-1)
        loss = cox_partial_likelihood_loss(risk, T_train, E_train)
        loss.backward(); optimizer.step()
        train_losses.append(float(loss.detach().item()))  # NEW
        model.eval(); risk_head.eval()
        with torch.no_grad():
            emb_v, _ = model(X_val, M_val); risk_v = risk_head(emb_v).squeeze(-1)
            val_loss = cox_partial_likelihood_loss(risk_v, T_val, E_val)
        val_losses.append(float(val_loss.item()))  # NEW
        if val_loss < best_loss:
            best_loss = val_loss; best_state = model.state_dict(); patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience: break
    if best_state: model.load_state_dict(best_state)
    from lifelines.utils import concordance_index
    with torch.no_grad():
        emb_v, _ = model(X_val, M_val); risk_v = risk_head(emb_v).squeeze(-1).cpu().numpy()
        ci = concordance_index(T_val.cpu().numpy(), -risk_v, E_val.cpu().numpy())
    # Return model + risk_head on CPU so downstream eval code (which uses CPU
    # tensors) keeps working without changes.
    model.to('cpu'); risk_head.to('cpu')
    best_epoch = int(np.argmin(val_losses)) if val_losses else 0  # NEW
    return {
        "best_val_cindex": ci,
        "risk_head": risk_head,
        "train_loss_history": train_losses,    # NEW
        "val_loss_history": val_losses,        # NEW
        "best_epoch": best_epoch,              # NEW
        "n_epochs_run": len(train_losses),     # NEW
    }

def benchmark_efficiency(model, sample_x, sample_m, n_warmup=10, n_runs=100) -> dict:
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup): _ = model(sample_x, sample_m)
        start = time.time()
        for _ in range(n_runs): _ = model(sample_x, sample_m)
        avg_latency = (time.time() - start) / n_runs * 1000
    import os, psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    params = sum(p.numel() for p in model.parameters())
    return {"latency_ms": avg_latency, "memory_mb": mem, "n_parameters": params}