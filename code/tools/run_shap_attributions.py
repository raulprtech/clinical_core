"""
SHAP attributions on phase_2_holdout artifacts.

Reproduces the paper's claim that high-capacity tabular models exploit the
post-event variables (ecog_score, karnofsky_score, tumor_status) much more
aggressively than the linear Cox model. The output is a per-feature bar
plot comparing mean |SHAP| between cox_baseline and ft_transformer under
both the limpio (19 vars) and permisivo (22 vars) protocols.

For the linear Cox baseline we use shap.LinearExplainer on the fitted
coefficients (exact, fast). For the FT-Transformer we use
shap.KernelExplainer on a small background subset — slower but model
agnostic, which avoids having to wire DeepExplainer to a custom forward.

Defaults:
  --seed 42         only one seed (KernelExplainer is expensive)
  --n-background 50 background samples drawn from the seed's TRAIN set
  --n-explain   100 held-out patients to attribute

Usage
-----
    python tools/run_shap_attributions.py <run_dir>
    python tools/run_shap_attributions.py <run_dir> --seed 42 --n-explain 80
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))


def _load_ckpt(run_dir: Path, protocol: str, seed: int, variant: str):
    p = run_dir / "phase2_artifacts" / "checkpoints" / f"{protocol}_seed{seed}_{variant}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"checkpoint missing: {p}")
    with open(p, 'rb') as f:
        return pickle.load(f)


def _load_pred(run_dir: Path, protocol: str, seed: int, variant: str):
    p = run_dir / "phase2_artifacts" / "predictions" / f"{protocol}_seed{seed}_{variant}.npz"
    if not p.exists():
        raise FileNotFoundError(f"predictions missing: {p}")
    return np.load(p, allow_pickle=True)


def _load_features(run_dir: Path) -> pd.DataFrame:
    """raw_features.csv carries the indexable feature matrix used by the run."""
    f = run_dir / "raw_features.csv"
    if not f.exists():
        raise FileNotFoundError(f"raw_features.csv missing in {run_dir}")
    return pd.read_csv(f, index_col=0)


def _slice_features(df_features: pd.DataFrame, case_ids: np.ndarray,
                    feature_cols: List[str]) -> np.ndarray:
    """
    Reconstruct the per-patient input to the model. Note we use the raw
    feature matrix (the values that fed the per-fold preprocessor). For the
    SHAP barplot the absolute magnitude is less important than the ranking
    across features.
    """
    str_ids = [str(c) for c in case_ids]
    sub = df_features.loc[[i for i in str_ids if i in df_features.index]]
    # column subset in the same order as the trained model expects
    cols = [c for c in feature_cols if c in sub.columns]
    sub = sub[cols].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).fillna(sub.median(numeric_only=True))
    sub = sub.fillna(0.0)
    return sub.values.astype(np.float64), cols


def _shap_cox(ckpt, X_explain, X_background, feature_cols):
    import shap
    cph = ckpt['cph']
    scaler = ckpt['scaler']
    coef = cph.params_.values  # log hazard ratios on standardized scale
    # Order of training columns is keep_var (== feature_cols in checkpoint)
    cox_cols = list(cph.params_.index)
    # Build X aligned to cox_cols
    df_explain = pd.DataFrame(X_explain, columns=feature_cols)
    df_bg = pd.DataFrame(X_background, columns=feature_cols)
    df_explain = df_explain.reindex(columns=cox_cols, fill_value=0.0)
    df_bg = df_bg.reindex(columns=cox_cols, fill_value=0.0)
    Xe = scaler.transform(df_explain.values)
    Xb = scaler.transform(df_bg.values)

    # LinearExplainer expects a model with `coef_`. We wrap it in a tiny shim.
    class _LinearModel:
        def __init__(self, coef):
            self.coef_ = coef
            self.intercept_ = 0.0
    explainer = shap.LinearExplainer(_LinearModel(coef), Xb,
                                     feature_perturbation="interventional")
    sv = explainer.shap_values(Xe)
    return sv, cox_cols


def _rebuild_ft_transformer(ckpt):
    from components.adapters.ingestion.tabular.models.ft_transformer import build_ft_transformer
    vk = dict(ckpt['variant_kwargs'])
    input_dim = int(vk.pop('input_dim'))
    output_dim = int(vk.pop('output_dim', 768))
    enc = build_ft_transformer(input_dim=input_dim, output_dim=output_dim, **vk)
    enc.load_state_dict(ckpt['encoder_state_dict'])
    risk_head = torch.nn.Linear(output_dim, 1)
    risk_head.load_state_dict(ckpt['risk_head_state_dict'])
    enc.eval(); risk_head.eval()
    return enc, risk_head


def _shap_ft(ckpt, X_explain, X_background, feature_cols, n_iter=100):
    """KernelExplainer on a model-agnostic forward."""
    import shap
    enc, risk_head = _rebuild_ft_transformer(ckpt)
    cols = list(ckpt['feature_cols'])
    # Reindex
    df_explain = pd.DataFrame(X_explain, columns=feature_cols).reindex(columns=cols, fill_value=0.0)
    df_bg = pd.DataFrame(X_background, columns=feature_cols).reindex(columns=cols, fill_value=0.0)
    Xe = df_explain.values.astype(np.float32)
    Xb = df_bg.values.astype(np.float32)

    def _predict(X_np):
        X_t = torch.tensor(X_np, dtype=torch.float32)
        # The model takes (x, mask); we pass all-present mask of ones (SHAP
        # background corresponds to "average patient", not missing data).
        mask = torch.ones_like(X_t)
        with torch.no_grad():
            emb, _ = enc(X_t, mask)
            risk = risk_head(emb).squeeze(-1).numpy()
        return risk

    explainer = shap.KernelExplainer(_predict, Xb)
    sv = explainer.shap_values(Xe, nsamples=n_iter, silent=True)
    return sv, cols


def _run_for_pair(run_dir, df_features, protocol, seed, n_explain, n_bg, n_iter):
    out = []
    for variant in ['cox_baseline', 'ft_transformer']:
        try:
            ckpt = _load_ckpt(run_dir, protocol, seed, variant)
            pred = _load_pred(run_dir, protocol, seed, variant)
        except FileNotFoundError as e:
            print(f"      skip {protocol}/{variant}: {e}")
            continue

        feature_cols = list(ckpt['feature_cols'])
        # explain subset = first n_explain held-out patients
        ho_ids = pred['case_ids_holdout'][:n_explain]
        tr_ids = pred['case_ids_train'][:n_bg]
        X_explain, used_cols = _slice_features(df_features, ho_ids, feature_cols)
        X_bg, _ = _slice_features(df_features, tr_ids, feature_cols)

        print(f"      → SHAP {protocol}/{variant} on n_explain={len(X_explain)}, n_bg={len(X_bg)}")
        if variant == 'cox_baseline':
            sv, cols_used = _shap_cox(ckpt, X_explain, X_bg, used_cols)
        else:
            sv, cols_used = _shap_ft(ckpt, X_explain, X_bg, used_cols, n_iter=n_iter)
        mean_abs = np.mean(np.abs(sv), axis=0)
        for c, v in zip(cols_used, mean_abs):
            out.append({
                'protocol': protocol, 'variant': variant, 'seed': int(seed),
                'feature': c, 'mean_abs_shap': float(v),
            })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--protocols", nargs='+', default=['limpio', 'permisivo'])
    ap.add_argument("--n-explain", type=int, default=100)
    ap.add_argument("--n-background", type=int, default=50)
    ap.add_argument("--n-iter", type=int, default=100,
                    help="KernelExplainer nsamples per held-out patient")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    print(f"[1/3] Loading raw_features.csv")
    df_features = _load_features(run_dir)

    print(f"[2/3] Computing SHAP for each protocol × variant (seed={args.seed})")
    all_rows = []
    for protocol in args.protocols:
        print(f"  protocol={protocol}")
        rows = _run_for_pair(run_dir, df_features, protocol, args.seed,
                             args.n_explain, args.n_background, args.n_iter)
        all_rows.extend(rows)

    if not all_rows:
        print("No SHAP rows generated — checkpoints likely missing.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    csv_out = run_dir / "phase2_shap_attributions.csv"
    df.to_csv(csv_out, index=False)
    print(f"      Wrote {csv_out}")

    print(f"[3/3] Comparative bar plot")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        leak_vars = {'ecog_score', 'karnofsky_score', 'tumor_status'}
        # Stable feature order: union over (protocol, variant), sorted by max SHAP
        order = (df.groupby('feature')['mean_abs_shap'].max()
                 .sort_values(ascending=False).index.tolist())

        protos = sorted(df['protocol'].unique())
        n_protos = len(protos)
        fig, axes = plt.subplots(1, n_protos, figsize=(7 * n_protos, max(5, 0.35 * len(order))),
                                 sharey=True)
        if n_protos == 1:
            axes = [axes]
        for ax, proto in zip(axes, protos):
            sub = df[df['protocol'] == proto]
            cox = sub[sub['variant'] == 'cox_baseline'].set_index('feature')['mean_abs_shap']
            ft  = sub[sub['variant'] == 'ft_transformer'].set_index('feature')['mean_abs_shap']
            cox = cox.reindex(order, fill_value=0)
            ft  = ft.reindex(order, fill_value=0)
            y = np.arange(len(order))
            ax.barh(y - 0.2, cox.values, height=0.4, label='Cox PH', color='#1f77b4')
            ax.barh(y + 0.2, ft.values,  height=0.4, label='FT-Transformer', color='#d62728')
            ax.set_yticks(y)
            colors = ['#d62728' if f in leak_vars else 'black' for f in order]
            ax.set_yticklabels(order, fontsize=8)
            for tick, col in zip(ax.get_yticklabels(), colors):
                tick.set_color(col)
            ax.invert_yaxis()
            ax.set_title(f'Protocolo: {proto}')
            ax.set_xlabel('Importancia (|SHAP| medio)')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(axis='x', alpha=0.3)
        fig.suptitle(
            'Atribuciones SHAP: Cox PH vs FT-Transformer\n'
            '(variables post-evento marcadas en rojo)',
            fontsize=11,
        )
        fig.tight_layout()
        png_out = run_dir / "phase2_shap_attributions.png"
        fig.savefig(png_out, dpi=150)
        print(f"      Wrote {png_out}")
    except Exception as e:
        print(f"      plot skipped: {e}")


if __name__ == "__main__":
    main()
