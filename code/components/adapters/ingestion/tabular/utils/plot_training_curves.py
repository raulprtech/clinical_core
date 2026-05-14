"""
plot_training_curves.py — diagnostic visualization for TABULAR-IN variants.

Reads the JSON files dumped by `_maybe_dump_curves` in experiment_runner.py
(located under `<run_dir>/training_curves/`) and produces:

    1. Per-variant aggregated curves (mean ± IQR over seeds and folds).
    2. Per-fold detail panels (one subplot per seed/fold) for the variant
       under diagnosis.
    3. A printed diagnostic verdict per variant:
         - "Healthy convergence" — train/val both decrease and plateau together.
         - "Overfitting"         — train keeps decreasing while val rises/stalls.
         - "Underfitting"        — both stay high; capacity or LR insufficient.
         - "Inconsistent"        — high variance across folds (cohort too small
                                   or initialization-sensitive).

Usage (from `code/`):
    python components/adapters/ingestion/tabular/utils/plot_training_curves.py \\
        --run_dir /path/to/results/<timestamp>_<hash> \\
        --out     /path/to/results/<timestamp>_<hash>/plots

Notes:
    • No additional dependencies beyond matplotlib + numpy.
    • If a variant has no JSON files (e.g. cox_baseline doesn't use
      train_variant_c), it is silently skipped.
    • Designed to be run AFTER the experiment has finished; non-destructive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ============================================================
# DATA LOADING
# ============================================================

def load_curves(run_dir: Path) -> Dict[str, List[dict]]:
    """
    Scan `<run_dir>/training_curves/*.json` and group by variant.

    Returns
    -------
    dict mapping variant_name -> list of payload dicts.
    Each payload has keys:
        seed, fold, variant, train_loss_history, val_loss_history,
        best_epoch, n_epochs_run, best_val_cindex.
    """
    curves_dir = run_dir / "training_curves"
    if not curves_dir.exists():
        raise FileNotFoundError(
            f"No training_curves directory at {curves_dir}. "
            "Did you apply the runner patch and run the experiment?"
        )

    grouped: Dict[str, List[dict]] = {}
    for json_path in sorted(curves_dir.glob("*.json")):
        with open(json_path) as f:
            payload = json.load(f)
        variant = payload.get('variant', json_path.stem.split('_')[-1])
        grouped.setdefault(variant, []).append(payload)
    return grouped


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_runs(
    runs: List[dict],
    which: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad-and-stack loss histories from multiple folds/seeds.

    Different runs may end at different epochs (early stopping), so we pad
    with NaN and aggregate ignoring missing values.

    Parameters
    ----------
    runs  : list of payload dicts from load_curves
    which : 'train_loss_history' or 'val_loss_history'

    Returns
    -------
    epochs       : np.ndarray of shape [max_epochs]
    mean_loss    : np.ndarray of shape [max_epochs]  (NaN where < 2 runs survived)
    p25          : 25th percentile per epoch
    p75          : 75th percentile per epoch
    """
    histories = [r[which] for r in runs if r.get(which)]
    if not histories:
        return np.array([]), np.array([]), np.array([]), np.array([])

    max_len = max(len(h) for h in histories)
    stacked = np.full((len(histories), max_len), np.nan, dtype=np.float64)
    for i, h in enumerate(histories):
        stacked[i, :len(h)] = h

    epochs = np.arange(max_len)
    with np.errstate(invalid='ignore'):
        mean_loss = np.nanmean(stacked, axis=0)
        p25 = np.nanpercentile(stacked, 25, axis=0)
        p75 = np.nanpercentile(stacked, 75, axis=0)
    return epochs, mean_loss, p25, p75


# ============================================================
# DIAGNOSTIC VERDICT
# ============================================================

def diagnose_variant(runs: List[dict]) -> dict:
    """
    Heuristic diagnostic over all runs of a single variant.

    Signals computed per run, then median-aggregated across folds/seeds:
      - train_last, val_last  : median loss over the LAST 10% of epochs.
      - drop_train, drop_val  : (first-10%-median) - (last-10%-median).
      - gap = val_last - train_last  (>0 means val sits above train).
      - cindex_std            : std of `best_val_cindex` across runs.

    Verdict order (most specific first):
      • INCONSISTENT  — cindex_std > 0.04. Fold-to-fold variance too high to
                        trust the average; usually means cohort is too small
                        or initialization is unstable. Don't conclude anything
                        else until this is resolved.
      • UNDERFITTING  — drop_train < 0.1 AND drop_val < 0.1. The model is not
                        learning; loss barely moves. Capacity, LR, or epochs
                        insufficient.
      • OVERFITTING   — gap > 0.3. Val stays measurably above train at the
                        end. Train continues to win but generalization gap
                        persists or widens. Apply regularization.
      • HEALTHY       — None of the above.

    The 0.3 gap threshold and 0.04 std threshold are calibrated for
    Cox partial-likelihood loss on N≈350 train cases (TCGA-KIRC after CV
    split). Adjust if the loss scale or cohort size changes substantially.
    """
    train_lasts, train_firsts, val_lasts, val_firsts, cidx = [], [], [], [], []
    for r in runs:
        tr = r.get('train_loss_history', []) or []
        vl = r.get('val_loss_history',   []) or []
        if not tr or not vl:
            continue
        n = min(len(tr), len(vl))
        tail = max(1, n // 10)
        head = max(1, n // 10)
        train_firsts.append(np.median(tr[:head]))
        train_lasts.append(np.median(tr[-tail:]))
        val_firsts.append(np.median(vl[:head]))
        val_lasts.append(np.median(vl[-tail:]))
        ci = r.get('best_val_cindex')
        if ci is not None:
            cidx.append(ci)

    if not train_lasts:
        return {'verdict': 'NO_DATA'}

    gap        = float(np.median(val_lasts) - np.median(train_lasts))
    drop_train = float(np.median(train_firsts) - np.median(train_lasts))
    drop_val   = float(np.median(val_firsts) - np.median(val_lasts))
    cidx_std   = float(np.std(cidx)) if len(cidx) > 1 else 0.0
    cidx_mean  = float(np.mean(cidx)) if cidx else float('nan')

    if cidx_std > 0.04:
        verdict = 'INCONSISTENT'
    elif drop_train < 0.1 and drop_val < 0.1:
        verdict = 'UNDERFITTING'
    elif gap > 0.3:
        verdict = 'OVERFITTING'
    else:
        verdict = 'HEALTHY'

    return {
        'verdict':         verdict,
        'gap_val_minus_train':     round(gap, 4),
        'drop_train':              round(drop_train, 4),
        'drop_val':                round(drop_val, 4),
        'cindex_mean':             round(cidx_mean, 4),
        'cindex_std':              round(cidx_std, 4),
        'n_runs':                  len(train_lasts),
    }


# ============================================================
# PLOTTING — AGGREGATED PER VARIANT
# ============================================================

def plot_aggregated(
    grouped: Dict[str, List[dict]],
    out_path: Path,
) -> None:
    """
    One subplot per variant. Mean ± IQR band over seeds/folds.
    """
    variants = sorted(grouped.keys())
    n = len(variants)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, variant in zip(axes, variants):
        runs = grouped[variant]
        e_tr, m_tr, p25_tr, p75_tr = aggregate_runs(runs, 'train_loss_history')
        e_vl, m_vl, p25_vl, p75_vl = aggregate_runs(runs, 'val_loss_history')

        if len(m_tr) > 0:
            ax.plot(e_tr, m_tr, label='train (mean)', linewidth=1.8)
            ax.fill_between(e_tr, p25_tr, p75_tr, alpha=0.18, label='train IQR')

        if len(m_vl) > 0:
            ax.plot(e_vl, m_vl, label='val (mean)', linewidth=1.8)
            ax.fill_between(e_vl, p25_vl, p75_vl, alpha=0.18, label='val IQR')

        diag = diagnose_variant(runs)
        title = f"{variant}\n[{diag['verdict']}]  C-index = {diag['cindex_mean']:.4f} ± {diag['cindex_std']:.4f}  (n_runs={diag['n_runs']})"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cox partial-likelihood loss')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    fig.suptitle('TABULAR-IN — Aggregated training curves (mean ± IQR)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# PLOTTING — DETAIL PANELS PER FOLD
# ============================================================

def plot_per_fold(
    grouped: Dict[str, List[dict]],
    out_dir: Path,
) -> None:
    """
    One figure per variant, with a grid of subplots: rows = seeds, cols = folds.
    Each subplot shows train + val for ONE fold so fold-to-fold variance is
    visible.
    """
    for variant, runs in grouped.items():
        seeds = sorted({r['seed'] for r in runs})
        folds = sorted({r['fold'] for r in runs})
        if not seeds or not folds:
            continue

        n_rows, n_cols = len(seeds), len(folds)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.2 * n_cols, 2.6 * n_rows),
            sharex=False, sharey=False,
            squeeze=False,
        )

        index = {(r['seed'], r['fold']): r for r in runs}
        for i, seed in enumerate(seeds):
            for j, fold in enumerate(folds):
                ax = axes[i][j]
                run = index.get((seed, fold))
                if run is None:
                    ax.set_visible(False)
                    continue
                tr = run.get('train_loss_history', [])
                vl = run.get('val_loss_history', [])
                ep = np.arange(max(len(tr), len(vl)))
                ax.plot(np.arange(len(tr)), tr, label='train', linewidth=1.3)
                ax.plot(np.arange(len(vl)), vl, label='val',   linewidth=1.3)
                best_ep = run.get('best_epoch')
                if best_ep is not None and 0 <= best_ep < len(vl):
                    ax.axvline(best_ep, color='gray', linestyle='--', linewidth=0.8)
                ci = run.get('best_val_cindex', float('nan'))
                ax.set_title(f"seed={seed} fold={fold}  C={ci:.3f}", fontsize=9)
                ax.tick_params(labelsize=8)
                if i == 0 and j == 0:
                    ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.2)

        fig.suptitle(f'{variant} — per-fold training curves', fontsize=12, y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / f'curves_per_fold_{variant}.png', dpi=140, bbox_inches='tight')
        plt.close(fig)


# ============================================================
# TEXT VERDICT (printed to stdout AND saved to file)
# ============================================================

def write_verdict(grouped: Dict[str, List[dict]], out_path: Path) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("TABULAR-IN — Diagnostic verdict per variant")
    lines.append("=" * 72)
    for variant in sorted(grouped.keys()):
        diag = diagnose_variant(grouped[variant])
        lines.append(f"\n[{variant}]")
        for k, v in diag.items():
            lines.append(f"    {k:25s} {v}")
    lines.append("\n" + "=" * 72)
    lines.append("Verdict heuristics:")
    lines.append("  INCONSISTENT  : C-index std across folds > 0.04 (checked first)")
    lines.append("  UNDERFITTING  : both train and val drops < 0.1  (loss barely moves)")
    lines.append("  OVERFITTING   : gap_val_minus_train > 0.3       (val sits above train)")
    lines.append("  HEALTHY       : none of the above")
    lines.append("=" * 72)
    text = "\n".join(lines)
    print(text)
    out_path.write_text(text)
    return text


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Directory containing training_curves/*.json')
    parser.add_argument('--out', type=str, default=None,
                        help='Output directory for plots (default: <run_dir>/plots)')
    parser.add_argument('--variants', type=str, nargs='*', default=None,
                        help='Optional filter: only plot these variant names')
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out).resolve() if args.out else (run_dir / 'plots')
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = load_curves(run_dir)
    if args.variants:
        grouped = {k: v for k, v in grouped.items() if k in args.variants}

    if not grouped:
        print(f"No curves found in {run_dir / 'training_curves'}.")
        return

    print(f"Loaded {sum(len(v) for v in grouped.values())} runs across {len(grouped)} variants.")
    plot_aggregated(grouped, out_dir / 'curves_aggregated.png')
    plot_per_fold(grouped, out_dir)
    write_verdict(grouped, out_dir / 'verdict.txt')
    print(f"\nPlots and verdict written to: {out_dir}")


if __name__ == '__main__':
    main()