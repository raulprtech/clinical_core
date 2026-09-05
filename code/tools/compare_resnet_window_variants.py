"""Paired comparison of single-window and CT multi-window sequence results."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for candidate in (CODE_ROOT, TOOLS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evaluate_fixed_sequence_ensemble import per_fold_metrics  # noqa: E402
from evaluate_resnet_sequence_nested_cv import (  # noqa: E402
    clustered_patient_bootstrap,
    file_sha256,
)
from evaluate_sequence_factorial_ablation import pooled_repeat_metrics  # noqa: E402


SOURCE_MODELS = (
    "mamba",
    "attention",
    "ensemble_train_percentile50",
    "ensemble_raw50",
    "ensemble_train_z50",
)
KEYS = ["repeat", "fold", "case_id"]
OUTCOMES = ["survival_days", "event", "modality"]


def paired_predictions(
    baseline: pd.DataFrame, candidate: pd.DataFrame
) -> pd.DataFrame:
    required = set(KEYS + OUTCOMES) | {f"risk_{name}" for name in SOURCE_MODELS}
    for label, frame in (("baseline", baseline), ("candidate", candidate)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} predictions are missing {sorted(missing)}")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{label} predictions contain duplicate keys")
    merged = baseline.merge(
        candidate,
        on=KEYS,
        how="outer",
        suffixes=("_single", "_multiwindow"),
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("Window variants do not contain identical held-out rows")
    for column in OUTCOMES:
        left = merged[f"{column}_single"]
        right = merged[f"{column}_multiwindow"]
        if column == "survival_days":
            equal = np.allclose(left, right, rtol=0, atol=0)
        else:
            equal = left.equals(right)
        if not equal:
            raise ValueError(f"Window variants disagree on {column}")
        merged[column] = left
    output = merged[KEYS + OUTCOMES].copy()
    for name in SOURCE_MODELS:
        output[f"risk_single_{name}"] = merged[f"risk_{name}_single"]
        output[f"risk_multiwindow_{name}"] = merged[f"risk_{name}_multiwindow"]
    if not np.isfinite(output.filter(like="risk_").to_numpy()).all():
        raise ValueError("Paired risks must be finite")
    return output.sort_values(KEYS).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=84049)
    args = parser.parse_args()

    baseline_path = args.baseline_dir / "heldout_predictions.csv"
    candidate_path = args.candidate_dir / "heldout_predictions.csv"
    predictions = paired_predictions(
        pd.read_csv(baseline_path), pd.read_csv(candidate_path)
    )
    names = tuple(
        f"{variant}_{model}"
        for variant in ("single", "multiwindow")
        for model in SOURCE_MODELS
    )
    folds = per_fold_metrics(predictions, names)
    repeats = pooled_repeat_metrics(predictions, list(names))
    comparisons = tuple(
        (f"multiwindow_{model}", f"single_{model}") for model in SOURCE_MODELS
    )
    bootstrap_frames = []
    for index, subgroup in enumerate(("all", "CT", "MR")):
        subset = (
            predictions
            if subgroup == "all"
            else predictions[predictions["modality"] == subgroup]
        )
        boot = clustered_patient_bootstrap(
            subset,
            comparisons,
            args.bootstrap_iterations,
            args.random_state + index * 10000,
        )
        boot.insert(0, "subgroup", subgroup)
        bootstrap_frames.append(boot)
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)

    fold_deltas = []
    for model in SOURCE_MODELS:
        single = folds[folds["model"] == f"single_{model}"].sort_values(KEYS[:2])
        multi = folds[folds["model"] == f"multiwindow_{model}"].sort_values(KEYS[:2])
        if not single[KEYS[:2]].reset_index(drop=True).equals(
            multi[KEYS[:2]].reset_index(drop=True)
        ):
            raise ValueError("Per-fold keys differ between window variants")
        deltas = multi["cindex_all"].to_numpy() - single["cindex_all"].to_numpy()
        fold_deltas.append({
            "model": model,
            "mean_fold_delta": float(deltas.mean()),
            "wins": int((deltas > 0).sum()),
            "ties": int((deltas == 0).sum()),
            "losses": int((deltas < 0).sum()),
        })

    summary = {
        "status": "paired_single_vs_fixed_ct_multiwindow",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "n_cases": int(predictions["case_id"].nunique()),
            "n_events": int(
                predictions[predictions["repeat"] == predictions["repeat"].min()][
                    "event"
                ].sum()
            ),
            "outer_repeats": int(predictions["repeat"].nunique()),
        },
        "primary_comparison": (
            "multiwindow_ensemble_train_percentile50 vs "
            "single_ensemble_train_percentile50"
        ),
        "paired_comparisons": bootstrap.to_dict(orient="records"),
        "fold_wins": fold_deltas,
        "claim_boundary": (
            "Internal same-cohort representation ablation; fixed windows were "
            "predeclared before outcome evaluation, but external confirmation remains required."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    repeats.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_cluster_bootstrap.csv", index=False)
    predictions.to_csv(args.output_dir / "paired_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    provenance = {
        "baseline_predictions": str(baseline_path.resolve()),
        "baseline_predictions_sha256": file_sha256(baseline_path),
        "candidate_predictions": str(candidate_path.resolve()),
        "candidate_predictions_sha256": file_sha256(candidate_path),
        "bootstrap_iterations": args.bootstrap_iterations,
        "random_state": args.random_state,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
