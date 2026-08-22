"""Evaluate a fixed 50/50 ensemble of existing OOF sequence risks.

The primary ensemble averages within-outer-fold percentile ranks so that the
two Cox models contribute equally despite arbitrary risk scales. Raw-risk and
foldwise z-score averages are retained as prespecified scale sensitivities.
Rank/z-score harmonization uses the unlabeled held-out risk distribution and is
therefore a post-hoc transductive diagnostic, not a deployable estimate.
"""

from __future__ import annotations

import argparse
import hashlib
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

from evaluate_resnet_sequence_models import safe_cindex  # noqa: E402
from evaluate_resnet_sequence_nested_cv import clustered_patient_bootstrap  # noqa: E402
from evaluate_sequence_factorial_ablation import pooled_repeat_metrics  # noqa: E402


SOURCE_MAMBA = "risk_mamba_t64_posoff"
SOURCE_ATTENTION = "risk_attention_t32_posoff"
MODEL_NAMES = (
    "mamba",
    "attention",
    "ensemble_rank50",
    "ensemble_raw50",
    "ensemble_z50",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_fixed_ensembles(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "repeat", "fold", "case_id", "survival_days", "event", "modality",
        SOURCE_MAMBA, SOURCE_ATTENTION,
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing {sorted(missing)}")
    if predictions.duplicated(["repeat", "case_id"]).any():
        raise ValueError("Each repeat must contain one OOF row per patient")
    output = predictions.copy()
    groups = output.groupby(["repeat", "fold"], sort=False)
    output["risk_mamba"] = output[SOURCE_MAMBA]
    output["risk_attention"] = output[SOURCE_ATTENTION]
    rank_mamba = groups[SOURCE_MAMBA].rank(method="average", pct=True)
    rank_attention = groups[SOURCE_ATTENTION].rank(method="average", pct=True)
    output["risk_ensemble_rank50"] = 0.5 * (rank_mamba + rank_attention)
    output["risk_ensemble_raw50"] = 0.5 * (
        output[SOURCE_MAMBA] + output[SOURCE_ATTENTION]
    )
    mean_mamba = groups[SOURCE_MAMBA].transform("mean")
    mean_attention = groups[SOURCE_ATTENTION].transform("mean")
    std_mamba = groups[SOURCE_MAMBA].transform(lambda values: values.std(ddof=0))
    std_attention = groups[SOURCE_ATTENTION].transform(
        lambda values: values.std(ddof=0)
    )
    if (std_mamba <= 0).any() or (std_attention <= 0).any():
        raise ValueError("Foldwise z-score requires nonconstant risks")
    z_mamba = (output[SOURCE_MAMBA] - mean_mamba) / std_mamba
    z_attention = (output[SOURCE_ATTENTION] - mean_attention) / std_attention
    output["risk_ensemble_z50"] = 0.5 * (z_mamba + z_attention)
    risk_columns = [f"risk_{name}" for name in MODEL_NAMES]
    if not np.isfinite(output[risk_columns].to_numpy()).all():
        raise ValueError("Ensemble risks must be finite")
    return output


def per_fold_metrics(
    predictions: pd.DataFrame, model_names: tuple[str, ...] = MODEL_NAMES
) -> pd.DataFrame:
    rows = []
    for (repeat, fold), frame in predictions.groupby(["repeat", "fold"], sort=True):
        for name in model_names:
            row = {"repeat": int(repeat), "fold": int(fold), "model": name}
            for subgroup in ("all", "CT", "MR"):
                subset = frame if subgroup == "all" else frame[
                    frame["modality"] == subgroup
                ]
                row[f"n_{subgroup.lower()}"] = len(subset)
                row[f"events_{subgroup.lower()}"] = int(subset["event"].sum())
                row[f"cindex_{subgroup.lower()}"] = safe_cindex(
                    subset["survival_days"].to_numpy(dtype=np.float64),
                    subset[f"risk_{name}"].to_numpy(dtype=np.float64),
                    subset["event"].to_numpy(dtype=np.int64),
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=7049)
    args = parser.parse_args()

    predictions = add_fixed_ensembles(pd.read_csv(args.predictions))
    folds = per_fold_metrics(predictions)
    repeats = pooled_repeat_metrics(predictions, list(MODEL_NAMES))
    comparisons = (
        ("ensemble_rank50", "mamba"),
        ("ensemble_rank50", "attention"),
        ("ensemble_raw50", "mamba"),
        ("ensemble_z50", "mamba"),
    )
    bootstrap = clustered_patient_bootstrap(
        predictions, comparisons, args.bootstrap_iterations, args.random_state
    )
    results = {}
    for name in MODEL_NAMES:
        selected = repeats[repeats["configuration"] == name]
        results[name] = {
            subgroup: {
                "mean_pooled_repeat_cindex": float(
                    selected[f"cindex_{subgroup}"].mean()
                ),
                "std_across_repeats": float(
                    selected[f"cindex_{subgroup}"].std(ddof=1)
                ),
                "per_repeat": selected[f"cindex_{subgroup}"].tolist(),
                "n_cases_per_repeat": int(selected[f"n_{subgroup}"].iloc[0]),
                "events_per_repeat": int(selected[f"events_{subgroup}"].iloc[0]),
            }
            for subgroup in ("all", "ct", "mr")
        }
    summary = {
        "status": "fixed_50_50_sequence_ensemble_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "n_cases_per_repeat": int(repeats["n_all"].iloc[0]),
            "events_per_repeat": int(repeats["events_all"].iloc[0]),
            "n_ct_per_repeat": int(repeats["n_ct"].iloc[0]),
            "n_mr_per_repeat": int(repeats["n_mr"].iloc[0]),
        },
        "protocol": {
            "weights": {"mamba_t64_posoff": 0.5, "attention_t32_posoff": 0.5},
            "primary_harmonization": "within_outer_heldout_fold_percentile_rank",
            "sensitivities": ["raw_risk", "within_outer_heldout_fold_zscore"],
            "bootstrap_iterations": args.bootstrap_iterations,
            "random_state": args.random_state,
        },
        "results": results,
        "paired_comparisons": bootstrap.to_dict(orient="records"),
        "claim_boundary": (
            "Post-hoc transductive diagnostic. Rank and z-score harmonization "
            "use unlabeled held-out risk distributions; deployment requires "
            "a predeclared train-derived scaling rule and new validation."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    repeats.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_cluster_bootstrap.csv", index=False)
    predictions.to_csv(args.output_dir / "heldout_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = {
        "script": str(Path(__file__).resolve()),
        "source_predictions": str(args.predictions.resolve()),
        "source_predictions_sha256": file_sha256(args.predictions),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps({
        "results": results,
        "comparisons": summary["paired_comparisons"],
        "claim_boundary": summary["claim_boundary"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
