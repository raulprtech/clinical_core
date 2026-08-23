"""Verify that two ResNet sequence caches differ only in token features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--modality-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_dir / "manifest.csv").sort_values("case_id")
    candidate = pd.read_csv(args.candidate_dir / "manifest.csv").sort_values("case_id")
    baseline = baseline.reset_index(drop=True)
    candidate = candidate.reset_index(drop=True)
    if baseline["case_id"].tolist() != candidate["case_id"].tolist():
        raise ValueError("Caches do not contain identical ordered case IDs")
    if baseline["SeriesInstanceUID"].tolist() != candidate["SeriesInstanceUID"].tolist():
        raise ValueError("Caches do not use identical selected series")
    if baseline["token_count"].tolist() != candidate["token_count"].tolist():
        raise ValueError("Caches do not contain identical token counts")
    modalities = pd.read_csv(args.modality_manifest)
    modalities["case_id"] = modalities["case_id"].astype(str).str.upper()
    modality = modalities.set_index("case_id")["Modality"].str.upper()

    rows = []
    total_tokens = 0
    for case_id in baseline["case_id"]:
        with np.load(
            args.baseline_dir / "cases" / f"{case_id}.npz", allow_pickle=False
        ) as left, np.load(
            args.candidate_dir / "cases" / f"{case_id}.npz", allow_pickle=False
        ) as right:
            left_features = left["features"].astype(np.float32)
            right_features = right["features"].astype(np.float32)
            if left_features.shape != right_features.shape:
                raise ValueError(f"Feature shape mismatch for {case_id}")
            if not np.array_equal(left["positions"], right["positions"]):
                raise ValueError(f"Position mismatch for {case_id}")
            if not np.isfinite(right_features).all():
                raise ValueError(f"Nonfinite candidate features for {case_id}")
            denominator = np.linalg.norm(left_features, axis=1) * np.linalg.norm(
                right_features, axis=1
            )
            cosine = np.sum(left_features * right_features, axis=1) / denominator
            rows.append({
                "modality": modality.loc[case_id],
                "mean_cosine": float(cosine.mean()),
                "min_cosine": float(cosine.min()),
            })
            total_tokens += len(cosine)
    metrics = pd.DataFrame(rows)
    failures_path = args.candidate_dir / "failures.csv"
    failures = pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()
    summary = {
        "n_cases": int(len(metrics)),
        "n_tokens": int(total_tokens),
        "n_failures": int(len(failures)),
        "identical_case_ids": True,
        "identical_series": True,
        "identical_token_counts": True,
        "identical_positions": True,
        "all_candidate_features_finite": True,
        "by_modality": {
            name: {
                "n_cases": int(len(frame)),
                "mean_patient_mean_token_cosine": float(frame["mean_cosine"].mean()),
                "minimum_patient_mean_token_cosine": float(frame["mean_cosine"].min()),
                "minimum_token_cosine": float(frame["min_cosine"].min()),
            }
            for name, frame in metrics.groupby("modality", sort=True)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
