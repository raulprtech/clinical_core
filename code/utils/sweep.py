"""
sweep.py — Hyperparameter sweep for the linear encoder (Variant C of TABULAR-CONN).

Random search over an FPGA-viable subspace of hyperparameters. Each trial
becomes a regular experiment_runner run with its own hash and directory,
so the full provenance chain is preserved.

Persistence:
  The sweep is resumable across sessions. On startup it scans the results
  directory for previously completed trials (identified by experiment name
  prefix) and skips any whose parameter signature has already been evaluated.
  Interrupt the loop at any time and re-launch — it continues where it left off.

FPGA viability:
  By default, trials whose linear encoder estimation exceeds FPGA_FLOP_BUDGET
  are skipped at sample time, never executed. This keeps the search inside
  the design space that the Paper 3 hardware contribution can actually
  synthesize. To explore the unconstrained space (e.g., to characterize the
  precision/efficiency trade-off), pass --allow_non_fpga.

Usage (local PC or Colab):
    python sweep.py --config experiment_config_tabular_only.yaml --n_trials 30

The script intentionally has zero new dependencies. It only uses the
existing experiment_runner + registry + config infrastructure.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml

from experiment_runner import run_experiment


# ============================================================
# SEARCH SPACE
# ============================================================
# Constrained to FPGA-viable values for the linear encoder.
# output_dim is FIXED at 768 because it defines the ingestion contract
# with FUSION-PROC and is therefore not a search dimension.
SEARCH_SPACE = {
    'hidden_dim':   [32, 64, 128, 256],
    'lr':           [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    'weight_decay': [0.0, 1e-5, 1e-4, 1e-3],
    'epochs':       [100, 200, 400],
    'patience':     [10, 20, 40],
}

# Approximate FLOP budget for one inference of the encoder when synthesized
# to FPGA. Calibrated against typical resources of mid-range FPGAs (e.g., the
# parts targeted by the Paper 3 contribution). Trials whose estimated FLOPs
# exceed this budget are excluded from the search by default.
FPGA_FLOP_BUDGET = 500_000


def estimate_flops(hidden_dim: int, input_dim: int = 20, output_dim: int = 768) -> int:
    """
    Approximate FLOPs for one forward pass of Variant C.
    
    Architecture: input → Linear → ReLU → Linear → LayerNorm → output
    """
    flops_l1 = 2 * input_dim * hidden_dim
    flops_relu = hidden_dim
    flops_l2 = 2 * hidden_dim * output_dim
    flops_ln = 4 * output_dim
    return flops_l1 + flops_relu + flops_l2 + flops_ln


def is_fpga_viable(trial_params: dict, input_dim: int = 20) -> bool:
    return estimate_flops(trial_params['hidden_dim'], input_dim=input_dim) <= FPGA_FLOP_BUDGET


# ============================================================
# TRIAL GENERATION
# ============================================================

def sample_trial(rng: random.Random) -> dict:
    """Sample one set of hyperparameters from the search space."""
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def trial_signature(trial: dict) -> str:
    """Deterministic short hash of trial parameters — used to detect duplicates."""
    canonical = json.dumps(trial, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ============================================================
# TRIAL CONFIG BUILDER
# ============================================================

def build_trial_config(base_config_path: Path, trial_params: dict, trial_id: int) -> dict:
    """
    Construct an experiment config for one trial: starts from the base config,
    forces phase 2 with only the linear_compact variant, and overrides its
    hyperparameters with the trial values. Other phases are disabled to
    minimize trial runtime.
    """
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    sig = trial_signature(trial_params)
    config['experiment']['name'] = f"sweep_variant_c_trial_{trial_id:04d}"
    config['experiment']['description'] = (
        f"Hyperparameter sweep for Variant C (linear encoder). "
        f"Trial {trial_id}, parameter signature {sig}, "
        f"params {trial_params}."
    )

    config['phase_2_variants']['variants'] = ['linear_compact']
    config['phase_2_variants']['variant_params'] = {
        'linear_compact': {
            'hidden_dim':   int(trial_params['hidden_dim']),
            'lr':           float(trial_params['lr']),
            'weight_decay': float(trial_params['weight_decay']),
            'epochs':       int(trial_params['epochs']),
            'patience':     int(trial_params['patience']),
        }
    }

    # Force explicit imputation strategies for both baseline and variants.
    # The base config may have 'auto' for variants, which depends on phase 1
    # running to pick a winner. Since phase 1 is disabled in sweep trials,
    # we must pin concrete strategies or every trial fails silently.
    # KNN k=5 is the winner documented in the protocol v12 preliminary results.
    config['phase_2_variants']['imputation_for_variants'] = 'knn_5'
    config['phase_2_variants']['imputation_for_baseline'] = 'mean_median'

    # Disable other phases — sweep cares only about Variant C performance
    for phase in ['phase_1_imputation', 'phase_3_efficiency',
                  'phase_4_stress', 'phase_5_multimodal']:
        if phase in config:
            config[phase]['enabled'] = False

    # Two seeds is a good compromise between trial cost and noise floor
    config['random']['seeds'] = [42, 123]

    return config


# ============================================================
# RESUME LOGIC
# ============================================================

def discover_completed_trials(results_dir: Path) -> Dict[str, dict]:
    """
    Scan results_dir for previously completed sweep trials.
    Returns {trial_signature: {trial_params, cindex_mean, run_dir, ...}}
    """
    completed: Dict[str, dict] = {}
    if not results_dir.exists():
        return completed

    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith('_'):
            continue

        config_file = run_dir / "experiment_config.yaml"
        summary_file = run_dir / "summary.json"
        if not (config_file.exists() and summary_file.exists()):
            continue

        try:
            with open(config_file) as f:
                cfg = yaml.safe_load(f)

            name = cfg.get('experiment', {}).get('name', '')
            if not name.startswith('sweep_variant_c'):
                continue

            params = cfg['phase_2_variants']['variant_params'].get('linear_compact', {})
            if not params:
                continue

            params_clean = {
                k: params[k] for k in
                ('hidden_dim', 'lr', 'weight_decay', 'epochs', 'patience')
                if k in params
            }
            sig = trial_signature(params_clean)

            with open(summary_file) as f:
                summary = json.load(f)

            ph2 = summary.get('phases', {}).get('phase_2', {})
            mean = ph2.get('mean', {}) if isinstance(ph2, dict) else {}
            cindex = mean.get('linear_compact')

            if cindex is not None:
                completed[sig] = {
                    'trial_params': params_clean,
                    'cindex_mean': float(cindex),
                    'run_dir': str(run_dir),
                    'name': name,
                }
        except Exception as e:
            print(f"  (warn) could not parse {run_dir.name}: {e}", file=sys.stderr)
            continue

    return completed


# ============================================================
# REPORTING
# ============================================================

def print_leaderboard(completed: dict, top_n: int = 10):
    valid = [
        (sig, info) for sig, info in completed.items()
        if info.get('cindex_mean') is not None
    ]
    valid.sort(key=lambda x: x[1]['cindex_mean'], reverse=True)

    if not valid:
        print("\nNo valid trials yet.")
        return

    print(f"\n{'=' * 80}")
    print(f"LEADERBOARD (top {min(top_n, len(valid))} of {len(valid)})")
    print(f"{'=' * 80}")
    header = (f"{'rank':<5}{'signature':<14}{'C-index':<10}"
              f"{'hidden':<8}{'lr':<10}{'wd':<10}{'epochs':<8}{'patience':<10}")
    print(header)
    print('-' * 80)

    for i, (sig, info) in enumerate(valid[:top_n], 1):
        p = info['trial_params']
        ci = info['cindex_mean']
        print(f"{i:<5}{sig:<14}{ci:<10.4f}"
              f"{p['hidden_dim']:<8}{p['lr']:<10.0e}{p['weight_decay']:<10.0e}"
              f"{p['epochs']:<8}{p['patience']:<10}")


def write_sweep_summary(results_dir: Path, completed: dict, fpga_only: bool):
    summary_path = results_dir / "_sweep_summary.json"
    valid = sorted(
        [{'signature': sig, **info} for sig, info in completed.items()
         if info.get('cindex_mean') is not None],
        key=lambda x: x['cindex_mean'], reverse=True
    )

    with open(summary_path, 'w') as f:
        json.dump({
            'updated_at': datetime.now().isoformat(),
            'n_trials_completed': len(valid),
            'fpga_constrained': fpga_only,
            'fpga_flop_budget': FPGA_FLOP_BUDGET,
            'search_space': SEARCH_SPACE,
            'top_10': valid[:10],
            'all_results': valid,
        }, f, indent=2, default=str)
    print(f"\nSweep summary written to: {summary_path}")


# ============================================================
# MAIN LOOP
# ============================================================

def run_sweep(
    base_config_path: str,
    n_trials: int = 30,
    seed: int = 0,
    fpga_only: bool = True,
    max_consecutive_duplicates: int = 50,
):
    base_config_path = Path(base_config_path).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    with open(base_config_path) as f:
        base_cfg = yaml.safe_load(f)

    results_dir = Path(base_cfg['output']['base_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SWEEP — Variant C hyperparameter search")
    print("=" * 80)
    print(f"Base config:    {base_config_path}")
    print(f"Results dir:    {results_dir}")
    print(f"Target trials:  {n_trials}")
    print(f"Sampling seed:  {seed}")
    print(f"FPGA-only:      {fpga_only}")
    print(f"FPGA budget:    {FPGA_FLOP_BUDGET:,} FLOPs")
    print()

    completed = discover_completed_trials(results_dir)
    print(f"Resumed: {len(completed)} previously completed trials found")

    n_done = len(completed)
    if n_done >= n_trials:
        print(f"Target of {n_trials} already reached. Showing leaderboard.")
        print_leaderboard(completed)
        write_sweep_summary(results_dir, completed, fpga_only)
        return completed

    print(f"Need to run {n_trials - n_done} additional trials")

    rng = random.Random(seed)
    # Burn through duplicate samples to reach the same point in the sequence
    # as if we had restarted from scratch — keeps the sweep deterministic.
    for _ in range(n_done * 3):  # rough heuristic to advance the rng
        sample_trial(rng)

    trial_id = n_done
    consecutive_dups = 0
    skipped_fpga = 0

    while n_done < n_trials:
        params = sample_trial(rng)
        sig = trial_signature(params)

        if sig in completed:
            consecutive_dups += 1
            if consecutive_dups > max_consecutive_duplicates:
                print(f"\nStopping: {consecutive_dups} consecutive duplicates — "
                      f"search space likely exhausted at this seed.")
                break
            continue

        if fpga_only and not is_fpga_viable(params):
            skipped_fpga += 1
            continue

        consecutive_dups = 0
        trial_id += 1
        flops = estimate_flops(params['hidden_dim'])

        print(f"\n[Trial {trial_id} / {n_trials}]  signature={sig}")
        print(f"  hidden_dim={params['hidden_dim']}, lr={params['lr']}, "
              f"wd={params['weight_decay']}, epochs={params['epochs']}, "
              f"patience={params['patience']}")
        print(f"  estimated FLOPs: {flops:,}")

        config = build_trial_config(base_config_path, params, trial_id)

        # Write trial config to a temp file next to the base config so that
        # any relative path inside the base config (e.g., feature_config)
        # still resolves correctly.
        tmp_path = base_config_path.parent / f".sweep_trial_{trial_id:04d}.yaml"
        with open(tmp_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)

        try:
            t0 = time.time()
            summary = run_experiment(str(tmp_path))
            elapsed = time.time() - t0

            ph2 = summary.get('phases', {}).get('phase_2', {})
            mean = ph2.get('mean', {}) if isinstance(ph2, dict) else {}
            cindex = mean.get('linear_compact')

            if cindex is None:
                print(f"  → run completed but no C-index reported")
                completed[sig] = {
                    'trial_params': params,
                    'cindex_mean': None,
                    'elapsed_seconds': elapsed,
                    'error': 'no_cindex',
                }
            else:
                print(f"  → C-index: {cindex:.4f}  ({elapsed:.1f}s)")
                completed[sig] = {
                    'trial_params': params,
                    'cindex_mean': float(cindex),
                    'elapsed_seconds': elapsed,
                }
            n_done += 1

            # Periodic checkpoint write so progress survives crashes
            if n_done % 5 == 0:
                write_sweep_summary(results_dir, completed, fpga_only)

        except Exception as e:
            print(f"  → FAILED: {e}")
            completed[sig] = {
                'trial_params': params,
                'cindex_mean': None,
                'error': str(e),
            }
            n_done += 1
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    print(f"\nSweep finished: {n_done} trials completed, {skipped_fpga} skipped (FPGA budget)")
    print_leaderboard(completed)
    write_sweep_summary(results_dir, completed, fpga_only)
    return completed


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for Variant C of TABULAR-CONN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--config', default='experiment_config_tabular_only.yaml',
        help='Base experiment config (default: experiment_config_tabular_only.yaml)'
    )
    parser.add_argument(
        '--n_trials', type=int, default=30,
        help='Target total number of trials (default: 30)'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random sampling seed (default: 0)'
    )
    parser.add_argument(
        '--allow_non_fpga', action='store_true',
        help='Allow trials that exceed the FPGA FLOP budget. '
             'Use only when characterizing the precision/efficiency trade-off.'
    )
    args = parser.parse_args()

    run_sweep(
        base_config_path=args.config,
        n_trials=args.n_trials,
        seed=args.seed,
        fpga_only=not args.allow_non_fpga,
    )


if __name__ == "__main__":
    main()