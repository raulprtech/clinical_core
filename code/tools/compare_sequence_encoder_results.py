"""Paired comparison of two frozen sequence-encoder survival evaluations."""

from __future__ import annotations

import argparse
import json
import re
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


def safe_label(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not result:
        raise ValueError("encoder labels must contain letters or digits")
    return result


def paired_predictions(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_label: str,
    candidate_label: str,
) -> pd.DataFrame:
    required = set(KEYS + OUTCOMES) | {f"risk_{name}" for name in SOURCE_MODELS}
    for label, frame in ((baseline_label, baseline), (candidate_label, candidate)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} predictions are missing {sorted(missing)}")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{label} predictions contain duplicate keys")
    merged = baseline.merge(
        candidate,
        on=KEYS,
        how="outer",
        suffixes=(f"_{baseline_label}", f"_{candidate_label}"),
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("Encoder variants do not contain identical held-out rows")
    for column in OUTCOMES:
        left = merged[f"{column}_{baseline_label}"]
        right = merged[f"{column}_{candidate_label}"]
        equal = (
            np.allclose(left, right, rtol=0, atol=0)
            if column == "survival_days"
            else left.equals(right)
        )
        if not equal:
            raise ValueError(f"Encoder variants disagree on {column}")
        merged[column] = left
    output = merged[KEYS + OUTCOMES].copy()
    for name in SOURCE_MODELS:
        output[f"risk_{baseline_label}_{name}"] = merged[
            f"risk_{name}_{baseline_label}"
        ]
        output[f"risk_{candidate_label}_{name}"] = merged[
            f"risk_{name}_{candidate_label}"
        ]
    if not np.isfinite(output.filter(like="risk_").to_numpy()).all():
        raise ValueError("Paired risks must be finite")
    return output.sort_values(KEYS).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=94049)
    args = parser.parse_args()

    baseline_label = safe_label(args.baseline_label)
    candidate_label = safe_label(args.candidate_label)
    if baseline_label == candidate_label:
        raise ValueError("encoder labels must differ")
    baseline_path = args.baseline_dir / "heldout_predictions.csv"
    candidate_path = args.candidate_dir / "heldout_predictions.csv"
    predictions = paired_predictions(
        pd.read_csv(baseline_path),
        pd.read_csv(candidate_path),
        baseline_label,
        candidate_label,
    )
    names = tuple(
        f"{encoder}_{model}"
        for encoder in (baseline_label, candidate_label)
        for model in SOURCE_MODELS
    )
    folds = per_fold_metrics(predictions, names)
    repeats = pooled_repeat_metrics(predictions, list(names))
    comparisons = tuple(
        (f"{candidate_label}_{model}", f"{baseline_label}_{model}")
        for model in SOURCE_MODELS
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
        left = folds[folds["model"] == f"{baseline_label}_{model}"].sort_values(
            KEYS[:2]
        )
        right = folds[folds["model"] == f"{candidate_label}_{model}"].sort_values(
            KEYS[:2]
        )
        if not left[KEYS[:2]].reset_index(drop=True).equals(
            right[KEYS[:2]].reset_index(drop=True)
        ):
            raise ValueError("Per-fold keys differ between encoder variants")
        deltas = right["cindex_all"].to_numpy() - left["cindex_all"].to_numpy()
        fold_deltas.append({
            "model": model,
            "mean_fold_delta": float(deltas.mean()),
            "wins": int((deltas > 0).sum()),
            "ties": int((deltas == 0).sum()),
            "losses": int((deltas < 0).sum()),
        })

    summary = {
        "status": "paired_frozen_sequence_encoder_comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_encoder": baseline_label,
        "candidate_encoder": candidate_label,
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
            f"{candidate_label}_ensemble_train_percentile50 vs "
            f"{baseline_label}_ensemble_train_percentile50"
        ),
        "paired_comparisons": bootstrap.to_dict(orient="records"),
        "fold_wins": fold_deltas,
        "claim_boundary": (
            "Internal same-cohort frozen representation ablation. Encoder "
            "weights and their prescribed normalization change together; "
            "external confirmation remains required."
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
