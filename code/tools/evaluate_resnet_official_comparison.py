"""Reproduce the official PCA+Cox baseline against saved sequence risks.

This post-benchmark evaluator reuses the exact outer split membership emitted
by evaluate_resnet_sequence_models.py. PCA, scaling and Cox regularization are
selected strictly inside each outer-train partition, then refitted on the full
outer-train before paired held-out comparison with Mamba.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for candidate in (CODE_ROOT, TOOLS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from components.processors.prognosis.utils.statistical_tests import (  # noqa: E402
    paired_cindex_test,
)
from evaluate_trimodal_fusion import (  # noqa: E402
    cindex,
    fit_predict,
    load_vision_csv,
    select_spec_and_oof,
)


DEFAULT_PCA_DIMS = [4, 8, 16, 24, 32, 48]
DEFAULT_PENALIZERS = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 20.0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_targets(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"case_id", "survival_days", "event"} - set(frame.columns)
    if missing:
        raise ValueError(f"Targets are missing columns: {sorted(missing)}")
    frame["case_id"] = frame["case_id"].astype(str).str.strip().str.upper()
    if frame["case_id"].duplicated().any():
        raise ValueError("Targets contain duplicate case_id values")
    frame["survival_days"] = pd.to_numeric(frame["survival_days"], errors="coerce")
    frame["event"] = pd.to_numeric(frame["event"], errors="coerce")
    return frame.set_index("case_id")[["survival_days", "event"]]


def validate_saved_benchmark(
    splits: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    split_required = {"seed", "case_id", "partition"}
    prediction_required = {
        "seed",
        "case_id",
        "risk_resnet18_2p5d_attention",
        "risk_resnet18_2p5d_mamba",
    }
    if not split_required.issubset(splits.columns):
        raise ValueError(f"Splits are missing {sorted(split_required - set(splits.columns))}")
    if not prediction_required.issubset(predictions.columns):
        raise ValueError(
            f"Predictions are missing {sorted(prediction_required - set(predictions.columns))}"
        )
    if splits.duplicated(["seed", "case_id"]).any():
        raise ValueError("Splits contain duplicate seed/case_id rows")
    if predictions.duplicated(["seed", "case_id"]).any():
        raise ValueError("Predictions contain duplicate seed/case_id rows")
    for seed, split in splits.groupby("seed"):
        heldout = set(split.loc[split["partition"] == "heldout", "case_id"])
        predicted = set(predictions.loc[predictions["seed"] == seed, "case_id"])
        if heldout != predicted:
            raise ValueError(f"Held-out predictions do not match split for seed {seed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-embeddings", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--benchmark-dir", required=True, type=Path)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--pca-dims", nargs="+", type=int, default=DEFAULT_PCA_DIMS)
    parser.add_argument(
        "--penalizers", nargs="+", type=float, default=DEFAULT_PENALIZERS
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    splits_path = args.benchmark_dir / "splits.csv"
    sequence_predictions_path = args.benchmark_dir / "heldout_predictions.csv"
    splits = pd.read_csv(splits_path)
    sequence_predictions = pd.read_csv(sequence_predictions_path)
    splits["case_id"] = splits["case_id"].astype(str).str.strip().str.upper()
    sequence_predictions["case_id"] = (
        sequence_predictions["case_id"].astype(str).str.strip().str.upper()
    )
    validate_saved_benchmark(splits, sequence_predictions)

    vision = load_vision_csv(args.baseline_embeddings)
    targets = load_targets(args.targets)
    prediction_rows = []
    metric_rows = []
    for seed, split in splits.groupby("seed", sort=True):
        train_ids = split.loc[split["partition"] != "heldout", "case_id"].tolist()
        heldout_ids = split.loc[split["partition"] == "heldout", "case_id"].tolist()
        case_ids = train_ids + heldout_ids
        missing = set(case_ids) - set(vision.index) | set(case_ids) - set(targets.index)
        if missing:
            raise ValueError(f"Seed {seed} has {len(missing)} missing baseline/target cases")
        values = vision.loc[case_ids].to_numpy(dtype=np.float64)
        survival = targets.loc[case_ids, "survival_days"].to_numpy(dtype=np.float64)
        events = targets.loc[case_ids, "event"].to_numpy(dtype=np.int64)
        train_idx = np.arange(len(train_ids))
        heldout_idx = np.arange(len(train_ids), len(case_ids))
        spec, _, inner_cindex = select_spec_and_oof(
            values[train_idx],
            survival[train_idx],
            events[train_idx],
            "vision",
            int(seed),
            args.pca_dims,
            args.penalizers,
            args.inner_folds,
        )
        _, official_risk = fit_predict(
            values,
            survival,
            events,
            train_idx,
            heldout_idx,
            "vision",
            spec,
            int(seed) * 1000 + 7,
        )
        saved = (
            sequence_predictions[sequence_predictions["seed"] == seed]
            .set_index("case_id")
            .loc[heldout_ids]
        )
        mamba_risk = saved["risk_resnet18_2p5d_mamba"].to_numpy(dtype=np.float64)
        attention_risk = saved[
            "risk_resnet18_2p5d_attention"
        ].to_numpy(dtype=np.float64)
        official_cindex = cindex(
            survival[heldout_idx], events[heldout_idx], official_risk
        )
        attention_cindex = cindex(
            survival[heldout_idx], events[heldout_idx], attention_risk
        )
        mamba_cindex = cindex(
            survival[heldout_idx], events[heldout_idx], mamba_risk
        )
        paired = paired_cindex_test(
            official_risk,
            mamba_risk,
            survival[heldout_idx],
            events[heldout_idx],
            n_iter=args.bootstrap_iterations,
            seed=int(seed) + 30000,
        )
        metric_rows.append({
            "seed": int(seed),
            "pca_dim": int(spec.pca_dim),
            "penalizer": float(spec.penalizer),
            "inner_cindex": float(inner_cindex),
            "cindex_official_pca_cox": official_cindex,
            "cindex_attention": attention_cindex,
            "cindex_mamba": mamba_cindex,
            "delta_mamba_minus_official": mamba_cindex - official_cindex,
            "delta_mamba_official_ci95_lo": paired["delta_lo"],
            "delta_mamba_official_ci95_hi": paired["delta_hi"],
            "delta_mamba_official_bootstrap_p": paired["p_value"],
        })
        prediction_rows.extend({
            "seed": int(seed),
            "case_id": case_id,
            "survival_days": float(survival[index]),
            "event": int(events[index]),
            "risk_official_pca_cox": float(official_risk[local]),
            "risk_attention": float(attention_risk[local]),
            "risk_mamba": float(mamba_risk[local]),
        } for local, (case_id, index) in enumerate(zip(heldout_ids, heldout_idx)))
        print(
            f"seed={seed} pca={spec.pca_dim} penalizer={spec.penalizer} "
            f"official={official_cindex:.4f} attention={attention_cindex:.4f} "
            f"mamba={mamba_cindex:.4f}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.benchmark_dir / "official_pca_cox_metrics.csv", index=False)
    metrics[[
        "seed",
        "cindex_official_pca_cox",
        "cindex_mamba",
        "delta_mamba_minus_official",
        "delta_mamba_official_ci95_lo",
        "delta_mamba_official_ci95_hi",
        "delta_mamba_official_bootstrap_p",
    ]].to_csv(
        args.benchmark_dir / "mamba_vs_official_paired_bootstrap.csv",
        index=False,
    )
    predictions.to_csv(
        args.benchmark_dir / "official_pca_cox_predictions.csv", index=False
    )
    model_columns = {
        "official_resnet18_2p5d_pca_cox": "cindex_official_pca_cox",
        "resnet18_2p5d_attention": "cindex_attention",
        "resnet18_2p5d_mamba": "cindex_mamba",
    }
    summary = {
        "status": "exploratory_paired_repeated_holdout",
        "cohort": {
            "n_cases": int(splits["case_id"].nunique()),
            "n_events": int(splits.groupby("case_id")["event"].first().sum()),
            "heldout_per_seed": int(predictions.groupby("seed").size().iloc[0]),
        },
        "protocol": {
            "official_baseline": "train-only StandardScaler + PCA + Cox ridge",
            "inner_folds": args.inner_folds,
            "pca_dims": args.pca_dims,
            "penalizers": args.penalizers,
            "bootstrap_iterations_per_seed": args.bootstrap_iterations,
            "same_outer_splits_as_sequence_models": True,
        },
        "models": {
            name: {
                "cindex_mean": float(metrics[column].mean()),
                "cindex_std_across_seeds": float(metrics[column].std(ddof=1)),
                "cindex_per_seed": metrics[column].tolist(),
            }
            for name, column in model_columns.items()
        },
        "mamba_minus_official": {
            "mean_delta": float(metrics["delta_mamba_minus_official"].mean()),
            "std_delta_across_seeds": float(
                metrics["delta_mamba_minus_official"].std(ddof=1)
            ),
            "wins": int((metrics["delta_mamba_minus_official"] > 0).sum()),
            "losses": int((metrics["delta_mamba_minus_official"] < 0).sum()),
            "individual_seed_bootstrap_ci_excludes_zero": int(
                (
                    (metrics["delta_mamba_official_ci95_lo"] > 0)
                    | (metrics["delta_mamba_official_ci95_hi"] < 0)
                ).sum()
            ),
        },
        "limitations": [
            "Repeated holdouts overlap and are not independent replications.",
            "No individual Mamba-minus-official bootstrap interval excludes zero.",
            "PCA+Cox refits on the full outer-train; neural models reserve validation for early stopping.",
            "External or nested repeated validation is required for a confirmatory claim.",
        ],
    }
    (args.benchmark_dir / "official_comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    provenance = {
        "baseline_embeddings": str(args.baseline_embeddings.resolve()),
        "baseline_sha256": file_sha256(args.baseline_embeddings),
        "targets": str(args.targets.resolve()),
        "targets_sha256": file_sha256(args.targets),
        "splits": str(splits_path.resolve()),
        "splits_sha256": file_sha256(splits_path),
        "sequence_predictions": str(sequence_predictions_path.resolve()),
        "sequence_predictions_sha256": file_sha256(sequence_predictions_path),
    }
    (args.benchmark_dir / "official_comparison_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
