"""
Post-hoc statistical-significance analysis for phase_2_holdout artifacts.

Reads the per-(protocol, seed, variant) prediction .npz files produced by
phase_2_holdout (with save_artifacts: true) and writes two CSVs into the
run directory:

  phase2_significance_ci.csv
      One row per (protocol, variant, seed) with the bootstrap mean C-index
      and its 95 % percentile interval, plus one row per (protocol, variant)
      pooled across seeds.

  phase2_significance_delta.csv
      Paired bootstrap test of the C-index difference between two protocols
      (default: limpio vs permisivo) on the same held-out patients of the
      same seed, for each variant. Reports observed delta, 95 % CI of the
      delta, and the two-sided bootstrap p-value.

The paired test is the operational analogue of the "DeLong test" referenced
in the manuscript: a same-patients comparison of two risk vectors. The
classical DeLong AUC statistic is not applicable to right-censored C-index,
so we use the bootstrap on the difference, which has the same null
hypothesis (no improvement) and the same paired structure.

Usage
-----
    python tools/run_significance_tests.py <run_dir>
    python tools/run_significance_tests.py <run_dir> --a permisivo --b limpio
    python tools/run_significance_tests.py <run_dir> --n-iter 2000 --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.analysis.statistical_tests import (
    bootstrap_cindex_ci,
    paired_cindex_test,
)


def _load_predictions(run_dir: Path) -> pd.DataFrame:
    pred_dir = run_dir / "phase2_artifacts" / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(
            f"No phase2_artifacts/predictions/ under {run_dir}. "
            "Re-run phase_2_holdout with save_artifacts: true."
        )
    rows = []
    for f in sorted(pred_dir.glob("*.npz")):
        # Expected key format: {protocol}_seed{N}_{variant}.npz
        stem = f.stem
        parts = stem.split("_seed")
        if len(parts) != 2:
            continue
        protocol = parts[0]
        rest = parts[1]
        seed_str, _, variant = rest.partition("_")
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        rows.append({'protocol': protocol, 'seed': seed, 'variant': variant, 'path': f})
    if not rows:
        raise RuntimeError(f"No usable .npz files under {pred_dir}")
    return pd.DataFrame(rows)


def _bootstrap_table(index_df: pd.DataFrame, n_iter: int, seed: int) -> pd.DataFrame:
    out = []
    for _, r in index_df.iterrows():
        d = np.load(r['path'], allow_pickle=True)
        point, mean_b, lo, hi = bootstrap_cindex_ci(
            d['risk_holdout'], d['times_holdout'], d['events_holdout'],
            n_iter=n_iter, seed=seed + int(r['seed']),
        )
        out.append({
            'protocol': r['protocol'], 'variant': r['variant'], 'seed': int(r['seed']),
            'cindex_point': point, 'cindex_bootstrap_mean': mean_b,
            'cindex_ci_lo': lo, 'cindex_ci_hi': hi,
            'n_holdout': int(len(d['times_holdout'])),
            'events_holdout': int(d['events_holdout'].sum()),
        })
    return pd.DataFrame(out)


def _pooled_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Pool across seeds: report mean over seeds + range of bootstrap CIs."""
    g = per_seed.groupby(['protocol', 'variant']).agg(
        cindex_mean=('cindex_point', 'mean'),
        cindex_std=('cindex_point', 'std'),
        ci_lo_min=('cindex_ci_lo', 'min'),
        ci_hi_max=('cindex_ci_hi', 'max'),
        n_seeds=('seed', 'count'),
    ).reset_index()
    return g.round(4)


def _paired_table(index_df: pd.DataFrame, proto_a: str, proto_b: str,
                  n_iter: int, seed: int) -> pd.DataFrame:
    """For each (variant, seed) compute paired delta proto_b minus proto_a."""
    out = []
    variants = sorted(index_df['variant'].unique())
    seeds = sorted(index_df['seed'].unique())
    for variant in variants:
        for s in seeds:
            row_a = index_df[(index_df['protocol'] == proto_a) &
                             (index_df['variant'] == variant) &
                             (index_df['seed'] == s)]
            row_b = index_df[(index_df['protocol'] == proto_b) &
                             (index_df['variant'] == variant) &
                             (index_df['seed'] == s)]
            if row_a.empty or row_b.empty:
                continue
            da = np.load(row_a.iloc[0]['path'], allow_pickle=True)
            db = np.load(row_b.iloc[0]['path'], allow_pickle=True)
            # Sanity: same held-out patients
            if not np.array_equal(da['case_ids_holdout'], db['case_ids_holdout']):
                # Align by case id intersection (defensive)
                common = np.intersect1d(da['case_ids_holdout'], db['case_ids_holdout'])
                if len(common) < 10:
                    continue
                idx_a = np.array([np.where(da['case_ids_holdout'] == c)[0][0] for c in common])
                idx_b = np.array([np.where(db['case_ids_holdout'] == c)[0][0] for c in common])
                risk_a = da['risk_holdout'][idx_a]
                risk_b = db['risk_holdout'][idx_b]
                time = da['times_holdout'][idx_a]
                event = da['events_holdout'][idx_a]
            else:
                risk_a = da['risk_holdout']
                risk_b = db['risk_holdout']
                time = da['times_holdout']
                event = da['events_holdout']

            res = paired_cindex_test(risk_a, risk_b, time, event,
                                     n_iter=n_iter, seed=seed + s)
            res.update({'variant': variant, 'seed': int(s),
                        'protocol_a': proto_a, 'protocol_b': proto_b})
            out.append(res)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("run_dir", type=Path, help="phase_2_holdout output directory")
    ap.add_argument("--a", default="limpio", help="reference protocol (default: limpio)")
    ap.add_argument("--b", default="permisivo", help="contrast protocol (default: permisivo)")
    ap.add_argument("--n-iter", type=int, default=1000, help="bootstrap iterations")
    ap.add_argument("--seed", type=int, default=0, help="base RNG seed")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"run_dir does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/3] Indexing predictions under {run_dir}")
    idx = _load_predictions(run_dir)
    print(f"      Found {len(idx)} prediction files across "
          f"{idx['protocol'].nunique()} protocols, "
          f"{idx['variant'].nunique()} variants, "
          f"{idx['seed'].nunique()} seeds.")

    print(f"[2/3] Bootstrap C-index 95% CI ({args.n_iter} iter / seed)")
    per_seed = _bootstrap_table(idx, n_iter=args.n_iter, seed=args.seed)
    pooled = _pooled_summary(per_seed)
    ci_out = run_dir / "phase2_significance_ci.csv"
    pd.concat([per_seed.assign(scope='per_seed'),
               pooled.assign(scope='pooled')], ignore_index=True).to_csv(ci_out, index=False)
    print(f"      Wrote {ci_out}")
    print(pooled.to_string(index=False))

    print(f"[3/3] Paired bootstrap test: {args.b} vs {args.a}")
    delta = _paired_table(idx, proto_a=args.a, proto_b=args.b,
                          n_iter=args.n_iter, seed=args.seed)
    if delta.empty:
        print("      No matched (protocol, seed, variant) pairs found.")
    else:
        delta_summary = (
            delta.groupby('variant')
            .agg(
                delta_mean_across_seeds=('delta_observed', 'mean'),
                delta_std_across_seeds=('delta_observed', 'std'),
                p_value_min_across_seeds=('p_value', 'min'),
                p_value_median=('p_value', 'median'),
                n_seeds=('seed', 'count'),
            ).round(5).reset_index()
        )
        delta_out = run_dir / "phase2_significance_delta.csv"
        pd.concat([delta.assign(scope='per_seed'),
                   delta_summary.assign(scope='pooled')], ignore_index=True).to_csv(delta_out, index=False)
        print(f"      Wrote {delta_out}")
        print(delta_summary.to_string(index=False))


if __name__ == "__main__":
    main()
