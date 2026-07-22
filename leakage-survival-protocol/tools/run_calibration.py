"""
Calibration analysis (integrated Brier + reliability curves) for the
artifacts written by phase_2_holdout with save_artifacts: true.

For each (protocol, seed, variant):
  - Cox baseline: lifelines survival function carried forward onto the
    common time grid.
  - Neural variants (linear_compact, ft_transformer): Breslow estimator
    fit on the saved training risk scores; the same estimator is then
    evaluated on the held-out risk scores.

The script writes:
  phase2_calibration_ibs.csv     One row per (protocol, variant, seed)
                                 with integrated Brier score.
  phase2_calibration_curves.csv  Reliability-diagram bins at 1/3/5-year
                                 horizons, one row per bin.
  phase2_calibration_curves.png  Cox vs Transformer reliability diagram
                                 (limpio vs permisivo, 3-year horizon).

Usage
-----
    python tools/run_calibration.py <run_dir>
    python tools/run_calibration.py <run_dir> --horizons 365 1095 1825
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.analysis.calibration import (
    cox_breslow_survival_for_neural,
    integrated_brier_score,
    calibration_curve_at_horizon,
)


def _index_predictions(run_dir: Path) -> pd.DataFrame:
    pred_dir = run_dir / "phase2_artifacts" / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(
            f"No phase2_artifacts/predictions/ under {run_dir}. "
            "Re-run phase_2_holdout with save_artifacts: true."
        )
    rows = []
    for f in sorted(pred_dir.glob("*.npz")):
        stem = f.stem
        parts = stem.split("_seed")
        if len(parts) != 2:
            continue
        protocol = parts[0]
        seed_str, _, variant = parts[1].partition("_")
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        rows.append({'protocol': protocol, 'seed': seed, 'variant': variant, 'path': f})
    if not rows:
        raise RuntimeError(f"No usable .npz files under {pred_dir}")
    return pd.DataFrame(rows)


def _survival_at_grid(npz, time_grid):
    """
    Recover S(t_grid | x_i) for one prediction file. Returns (n_grid, n_test).
    Cox saved a 'surv_func_holdout' matrix directly; neural saves only risks
    and we run Breslow here.
    """
    if 'surv_func_holdout' in npz.files:
        sf_times = npz['surv_time_grid']
        sf_vals = npz['surv_func_holdout']  # (n_native_times, n_test) from lifelines
        out = np.empty((len(time_grid), sf_vals.shape[1]), dtype=np.float64)
        for j, t in enumerate(time_grid):
            idx = np.searchsorted(sf_times, t, side='right') - 1
            if idx < 0:
                out[j, :] = 1.0
            else:
                out[j, :] = sf_vals[idx, :]
        return out
    # Neural: Breslow from training risks
    return cox_breslow_survival_for_neural(
        risk_train=npz['risk_train'],
        time_train=npz['times_train'],
        event_train=npz['events_train'],
        risk_test=npz['risk_holdout'],
        time_grid=time_grid,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--horizons", type=float, nargs="+",
                    default=[365.0, 1095.0, 1825.0],
                    help="Horizons in days (default: 1, 3, 5 years)")
    ap.add_argument("--ibs-grid-points", type=int, default=20)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    print(f"[1/4] Indexing predictions under {run_dir}")
    idx = _index_predictions(run_dir)
    print(f"      {len(idx)} files; protocols={sorted(idx['protocol'].unique())}, "
          f"variants={sorted(idx['variant'].unique())}, "
          f"seeds={sorted(idx['seed'].unique())}")

    print(f"[2/4] Integrated Brier score per (protocol, variant, seed)")
    ibs_rows = []
    for _, r in idx.iterrows():
        d = np.load(r['path'], allow_pickle=True)
        # IBS grid: span the held-out times, conservative.
        t_min = max(float(d['times_train'].min()), float(d['times_holdout'].min())) + 1.0
        t_max = min(float(d['times_train'].max()), float(d['times_holdout'].max())) - 1.0
        if t_max <= t_min:
            continue
        time_grid = np.linspace(t_min, t_max, args.ibs_grid_points)
        try:
            surv = _survival_at_grid(d, time_grid)
            ibs = integrated_brier_score(
                d['times_train'], d['events_train'],
                d['times_holdout'], d['events_holdout'],
                surv, time_grid,
            )
        except Exception as e:
            print(f"      WARN {r['protocol']}/{r['variant']}/seed{r['seed']}: {e}")
            ibs = float('nan')
        ibs_rows.append({
            'protocol': r['protocol'], 'variant': r['variant'], 'seed': int(r['seed']),
            'ibs': float(ibs),
            't_min': float(t_min), 't_max': float(t_max),
        })
    ibs_df = pd.DataFrame(ibs_rows)
    ibs_out = run_dir / "phase2_calibration_ibs.csv"
    ibs_df.to_csv(ibs_out, index=False)
    ibs_summary = (
        ibs_df.groupby(['protocol', 'variant'])['ibs']
        .agg(['mean', 'std', 'count']).round(4).reset_index()
    )
    print(f"      Wrote {ibs_out}")
    print(ibs_summary.to_string(index=False))

    print(f"[3/4] Reliability curves at {args.horizons} days")
    curve_rows = []
    for _, r in idx.iterrows():
        d = np.load(r['path'], allow_pickle=True)
        for h in args.horizons:
            try:
                surv_at_h = _survival_at_grid(d, np.array([h]))[0]
                curve = calibration_curve_at_horizon(
                    d['times_holdout'], d['events_holdout'], surv_at_h,
                    horizon=h, n_bins=10,
                )
                for mid, obs, n in zip(curve['bin_mid_pred'], curve['observed_surv'],
                                       curve['n_per_bin']):
                    curve_rows.append({
                        'protocol': r['protocol'], 'variant': r['variant'],
                        'seed': int(r['seed']), 'horizon': float(h),
                        'predicted_surv_mid': float(mid),
                        'observed_surv': float(obs),
                        'n_in_bin': int(n),
                    })
            except Exception as e:
                print(f"      WARN {r['protocol']}/{r['variant']}/seed{r['seed']} @ {h}: {e}")
    curve_df = pd.DataFrame(curve_rows)
    curve_out = run_dir / "phase2_calibration_curves.csv"
    curve_df.to_csv(curve_out, index=False)
    print(f"      Wrote {curve_out}")

    print(f"[4/4] Reliability diagram PNG (Cox vs FT-Transformer @ 3y, both protocols)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        target_horizon = 1095.0
        target_h = min(args.horizons, key=lambda x: abs(x - target_horizon))
        sub = curve_df[curve_df['horizon'] == target_h]
        if sub.empty:
            print("      no rows at requested horizon, skipping plot")
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
            styles = {
                ('limpio', 'cox_baseline'):     dict(color='#1f77b4', marker='o', ls='-'),
                ('permisivo', 'cox_baseline'):  dict(color='#1f77b4', marker='s', ls='--'),
                ('limpio', 'ft_transformer'):   dict(color='#d62728', marker='o', ls='-'),
                ('permisivo', 'ft_transformer'):dict(color='#d62728', marker='s', ls='--'),
            }
            for (proto, variant), st in styles.items():
                cell = sub[(sub['protocol'] == proto) & (sub['variant'] == variant)]
                if cell.empty:
                    continue
                agg = (cell.groupby(pd.cut(cell['predicted_surv_mid'],
                                          bins=np.linspace(0, 1, 11)),
                                    observed=False)
                       .agg(pred=('predicted_surv_mid', 'mean'),
                            obs=('observed_surv', 'mean'))
                       .dropna())
                ax.plot(agg['pred'], agg['obs'], label=f"{variant} ({proto})", **st)
            ax.set_xlabel(f'Predicted S(t={int(target_h)} d | x)')
            ax.set_ylabel(f'Observed S(t={int(target_h)} d) [KM]')
            ax.set_title('Calibración: Cox vs FT-Transformer')
            ax.legend(loc='lower right', fontsize=9)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            png_out = run_dir / "phase2_calibration_curves.png"
            fig.tight_layout()
            fig.savefig(png_out, dpi=150)
            print(f"      Wrote {png_out}")
    except Exception as e:
        print(f"      plot skipped: {e}")


if __name__ == "__main__":
    main()
