"""
validate_top3.py — Robust 5-seed validation of the top sweep candidates.

The sweep identified the 3 best hyperparameter configurations with single-seed
trials. This script re-runs those 3 configurations with 5 seeds each to verify
that the ranking is stable and to estimate true mean ± std of C-index.

Why 5 seeds (not 2, not 10):
  - 2 seeds is the minimum to detect a total-failure outlier but gives no real
    estimate of variance.
  - 5 seeds is the literature standard for survival-analysis benchmarks and
    gives a defensible std estimate with a paired t-test possible.
  - 10 seeds is overkill for this scale of model and these data volumes.

Output:
  results/_validation_top3.json — structured summary with per-seed C-index
  for each candidate, overall mean ± std, and a stability verdict.

Usage:
    python validate_top3.py --config experiment_config_tabular_only.yaml
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import yaml

from experiment_runner import run_experiment


# ============================================================
# THE 3 CANDIDATES TO VALIDATE
# ============================================================
# These are the top-3 from the single-seed sweep documented in
# _sweep_summary.json (2026-04-16 run). Ranked by single-seed C-index.
CANDIDATES = [
    {
        'rank_from_sweep': 1,
        'signature': 'f288d4fe7438',
        'params': {
            'hidden_dim':   64,
            'lr':           0.01,
            'weight_decay': 1e-05,
            'epochs':       200,
            'patience':     10,
        },
        'cindex_single_seed': 0.8130,
    },
    {
        'rank_from_sweep': 2,
        'signature': 'd37c55d681ff',
        'params': {
            'hidden_dim':   64,
            'lr':           0.01,
            'weight_decay': 0.001,
            'epochs':       400,
            'patience':     20,
        },
        'cindex_single_seed': 0.8113,
    },
    {
        'rank_from_sweep': 3,
        'signature': '3562004a3859',
        'params': {
            'hidden_dim':   128,
            'lr':           0.01,
            'weight_decay': 1e-05,
            'epochs':       200,
            'patience':     10,
        },
        'cindex_single_seed': 0.8093,
    },
]

# Five seeds. Fixed list for reproducibility. The first is the same seed
# the sweep used so we can compare directly seed-to-seed.
SEEDS = [42, 123, 7, 314, 1618]


# ============================================================
# CONFIG BUILDER (one config per candidate, 5 seeds each)
# ============================================================

def build_validation_config(base_config_path: Path, candidate: dict) -> dict:
    """Build an experiment_runner config for one candidate with 5 seeds."""
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    # Resolve feature_config relative to base_config_path IF it's relative.
    # This prevents resolution errors when we run from different CWDs.
    feat_path = Path(config['data']['feature_config'])
    if not feat_path.is_absolute():
        config['data']['feature_config'] = str((base_config_path.parent / feat_path).resolve())

    sig = candidate['signature']
    config['experiment']['name'] = f"validate_top3_{sig}"
    config['experiment']['description'] = (
        f"5-seed validation of sweep rank #{candidate['rank_from_sweep']} "
        f"(signature {sig}, single-seed C-index = {candidate['cindex_single_seed']}). "
        f"Hyperparameters: {candidate['params']}."
    )

    # Same contract as the sweep: only linear_fpga, pinned imputation.
    config['phase_2_variants']['variants'] = ['linear_fpga']
    config['phase_2_variants']['variant_params'] = {
        'linear_fpga': {
            'hidden_dim':   int(candidate['params']['hidden_dim']),
            'lr':           float(candidate['params']['lr']),
            'weight_decay': float(candidate['params']['weight_decay']),
            'epochs':       int(candidate['params']['epochs']),
            'patience':     int(candidate['params']['patience']),
        }
    }
    config['phase_2_variants']['imputation_for_variants'] = 'knn_5'
    config['phase_2_variants']['imputation_for_baseline'] = 'mean_median'

    for phase in ['phase_1_imputation', 'phase_3_efficiency',
                  'phase_4_stress', 'phase_5_multimodal']:
        if phase in config:
            config[phase]['enabled'] = False

    # The core of this script: 5 seeds instead of 1 or 2.
    config['random']['seeds'] = SEEDS

    return config


# ============================================================
# RESULT PARSING
# ============================================================

def extract_per_seed_cindex(run_dir: Path) -> Dict[int, float]:
    """
    Read phase_2 output and return {seed: cindex} for the linear_fpga variant.
    Falls back to the aggregated mean if per-seed breakdown is unavailable.
    """
    summary_path = run_dir / 'summary.json'
    if not summary_path.exists():
        return {}

    with open(summary_path) as f:
        summary = json.load(f)

    # experiment_runner.py writes phase2_variants.csv in the run directory.
    # It contains columns: seed, fold, variant, cindex, etc.
    csv_path = run_dir / 'phase2_variants.csv'
    if not csv_path.exists():
        return {}

    try:
        df = pd.read_csv(csv_path)
        # Filter for the variant we care about
        df_v = df[df['variant'] == 'linear_fpga']
        if df_v.empty:
            return {}
        
        # Group by seed and take the mean (across folds)
        by_seed = df_v.groupby('seed')['cindex'].mean().to_dict()
        return {int(s): float(c) for s, c in by_seed.items()}
    except Exception as e:
        # Fallback if CSV is malformed
        return {}

    return {}


def extract_overall_cindex(run_dir: Path) -> Optional[float]:
    """Read the final aggregated C-index for linear_fpga across all seeds."""
    summary_path = run_dir / 'summary.json'
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    # experiment_runner structure: summary['phases']['phase_2']['mean']['linear_fpga']
    phases = summary.get('phases', {})
    ph2 = phases.get('phase_2') or phases.get('phase_2_variants') or {}
    
    if isinstance(ph2, dict):
        # Check 'mean' sub-dict
        mean_dict = ph2.get('mean', {})
        if 'linear_fpga' in mean_dict:
            return float(mean_dict['linear_fpga'])
        
        # Check direct keys (fallback)
        for k in ('cindex_mean', 'c_index_mean', 'cindex', 'c_index'):
            if k in ph2:
                return float(ph2[k])
            
    return None
    return None


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='Base experiment config (same as sweep)')
    parser.add_argument('--results_dir', default='results',
                        help='Directory to persist per-candidate runs and final summary')
    args = parser.parse_args()

    base_config_path = Path(args.config)
    if not base_config_path.exists():
        print(f"ERROR: base config not found at {base_config_path}")
        sys.exit(1)

    with open(base_config_path) as f:
        base_cfg = yaml.safe_load(f)

    # Use results_dir from config if not provided in CLI
    # This ensures consistency with experiment_runner.py and sweep.py
    if args.results_dir == 'results' and 'output' in base_cfg:
        results_dir = Path(base_cfg['output']['base_dir'])
    else:
        results_dir = Path(args.results_dir)
    
    results_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("CLINICAL-CORE / TABULAR-IN — Top-3 Robust Validation")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print(f"Candidates to validate: {len(CANDIDATES)}")
    print()

    validation_results = []
    t_start = time.time()

    for i, candidate in enumerate(CANDIDATES, start=1):
        sig = candidate['signature']
        single_seed_cindex = candidate['cindex_single_seed']

        print(f"[{i}/{len(CANDIDATES)}] Candidate {sig} "
              f"(rank #{candidate['rank_from_sweep']}, single-seed = {single_seed_cindex})")
        print(f"  params: {candidate['params']}")

        config = build_validation_config(base_config_path, candidate)
        t_trial = time.time()
        try:
            summary = run_experiment(config)
            run_dir = summary['run_dir']
        except Exception as e:
            print(f"  → FAILED: {type(e).__name__}: {e}")
            validation_results.append({
                'candidate': candidate,
                'status': 'failed',
                'error': f"{type(e).__name__}: {e}",
                'elapsed_seconds': time.time() - t_trial,
            })
            continue

        run_dir = Path(run_dir)
        per_seed = extract_per_seed_cindex(run_dir)
        overall = extract_overall_cindex(run_dir)

        if per_seed:
            seed_values = [per_seed[s] for s in SEEDS if s in per_seed]
            mean_cindex = statistics.mean(seed_values) if seed_values else None
            std_cindex  = statistics.stdev(seed_values) if len(seed_values) > 1 else 0.0
            min_cindex  = min(seed_values) if seed_values else None
            max_cindex  = max(seed_values) if seed_values else None
        else:
            # No per-seed data available; fall back to overall mean only.
            seed_values = []
            mean_cindex = overall
            std_cindex = None
            min_cindex = max_cindex = None

        print(f"  per-seed C-index: {per_seed}")
        if std_cindex is not None:
            print(f"  → mean ± std over {len(seed_values)} seeds: "
                  f"{mean_cindex:.4f} ± {std_cindex:.4f}")
            print(f"  → range: [{min_cindex:.4f}, {max_cindex:.4f}]")
            drift = mean_cindex - single_seed_cindex
            print(f"  → drift from single-seed estimate: {drift:+.4f}")
        elif mean_cindex is not None:
            print(f"  → overall mean (per-seed unavailable): {mean_cindex:.4f}")
        else:
            print(f"  → no C-index recovered from run_dir={run_dir}")
        print(f"  elapsed: {time.time() - t_trial:.1f}s")
        print()

        validation_results.append({
            'candidate': candidate,
            'status': 'completed',
            'run_dir': str(run_dir),
            'per_seed_cindex': per_seed,
            'cindex_mean_5seeds': mean_cindex,
            'cindex_std_5seeds': std_cindex,
            'cindex_min_5seeds': min_cindex,
            'cindex_max_5seeds': max_cindex,
            'cindex_single_seed_from_sweep': single_seed_cindex,
            'drift_from_single_seed': (mean_cindex - single_seed_cindex
                                       if mean_cindex is not None else None),
            'elapsed_seconds': time.time() - t_trial,
        })

    # ------------------------------------------------------------
    # Stability verdict
    # ------------------------------------------------------------
    completed = [r for r in validation_results if r['status'] == 'completed'
                 and r.get('cindex_mean_5seeds') is not None]

    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    if not completed:
        print("No candidate completed successfully. Investigate the runs.")
    else:
        # Re-rank by 5-seed mean and compare to sweep order
        reranked = sorted(completed,
                          key=lambda r: r['cindex_mean_5seeds'],
                          reverse=True)
        print(f"{'Rank':<6}{'Sig':<16}{'Sweep#':<8}{'Mean±Std':<22}{'Range':<22}{'Drift':<8}")
        print("-" * 82)
        for new_rank, r in enumerate(reranked, start=1):
            sig   = r['candidate']['signature']
            swp   = r['candidate']['rank_from_sweep']
            mean  = r['cindex_mean_5seeds']
            std   = r['cindex_std_5seeds']
            lo    = r.get('cindex_min_5seeds')
            hi    = r.get('cindex_max_5seeds')
            drift = r['drift_from_single_seed']
            std_s  = f"{std:.4f}" if std is not None else "n/a"
            range_s = (f"[{lo:.4f}, {hi:.4f}]"
                       if lo is not None and hi is not None else "n/a")
            drift_s = f"{drift:+.4f}" if drift is not None else "n/a"
            print(f"{new_rank:<6}{sig:<16}#{swp:<7}{mean:.4f} ± {std_s:<10}"
                  f"{range_s:<22}{drift_s:<8}")
        print()

        top_5seed = reranked[0]
        top_sweep = next(r for r in completed
                         if r['candidate']['rank_from_sweep'] == 1)
        if top_5seed['candidate']['signature'] == top_sweep['candidate']['signature']:
            print(f"✓ STABILITY: sweep ranking preserved. Winner is still "
                  f"{top_5seed['candidate']['signature']} at "
                  f"{top_5seed['cindex_mean_5seeds']:.4f} ± "
                  f"{top_5seed['cindex_std_5seeds']:.4f}.")
        else:
            print(f"⚠ STABILITY: ranking CHANGED. "
                  f"5-seed winner is {top_5seed['candidate']['signature']} "
                  f"(sweep rank #{top_5seed['candidate']['rank_from_sweep']}), "
                  f"not the sweep winner.")

    # ------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------
    summary_path = results_dir / '_validation_top3.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'updated_at': datetime.utcnow().isoformat(),
            'seeds': SEEDS,
            'elapsed_total_seconds': time.time() - t_start,
            'results': validation_results,
        }, f, indent=2)
    print(f"\nValidation summary written to: {summary_path}")


if __name__ == '__main__':
    main()
