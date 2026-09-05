"""Paired nested-CV test of STU-Net pooling inside trimodal fusion.

Historical STU-Net regional means and renal volumetric moments are evaluated on
identical patients, outer folds and train-only tabular/text models. For each
visual representation, PCA/Cox hyperparameters and convex fusion weights are
selected exclusively inside the corresponding outer-train partition.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


CODE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for candidate in (CODE_ROOT, TOOLS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evaluate_resnet_sequence_nested_cv import (  # noqa: E402
    clustered_patient_bootstrap,
    file_sha256,
)
from evaluate_trimodal_fusion import (  # noqa: E402
    cindex,
    empirical_percentile,
    fit_predict,
    load_indexed_csv,
    load_text_npz,
    load_vision_csv,
    select_fusion_weights,
    select_spec_and_oof,
)


MODEL_NAMES = (
    "tabular",
    "text",
    "vision_mean",
    "vision_moments",
    "fusion_mean",
    "fusion_moments",
    "fusion_bimodal_moments",
)


def repeat_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for repeat, frame in predictions.groupby("repeat", sort=True):
        if frame["case_id"].duplicated().any():
            raise ValueError(f"Repeat {repeat} contains duplicate OOF patients")
        row: dict[str, Any] = {
            "repeat": int(repeat),
            "n_cases": len(frame),
            "n_events": int(frame["event"].sum()),
        }
        for name in MODEL_NAMES:
            row[f"cindex_{name}"] = cindex(
                frame["survival_days"].to_numpy(dtype=np.float64),
                frame["event"].to_numpy(dtype=np.int64),
                frame[f"risk_{name}"].to_numpy(dtype=np.float64),
            )
        row["delta_vision_moments_minus_mean"] = (
            row["cindex_vision_moments"] - row["cindex_vision_mean"]
        )
        row["delta_fusion_moments_minus_mean"] = (
            row["cindex_fusion_moments"] - row["cindex_fusion_mean"]
        )
        row["delta_fusion_moments_minus_tabular"] = (
            row["cindex_fusion_moments"] - row["cindex_tabular"]
        )
        row["delta_bimodal_minus_trimodal_moments"] = (
            row["cindex_fusion_bimodal_moments"]
            - row["cindex_fusion_moments"]
        )
        row["delta_bimodal_minus_tabular"] = (
            row["cindex_fusion_bimodal_moments"] - row["cindex_tabular"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def fit_outer_modality(
    values: np.ndarray,
    survival: np.ndarray,
    events: np.ndarray,
    outer_train: np.ndarray,
    heldout: np.ndarray,
    kind: str,
    split_seed: int,
    pca_dims: list[int],
    penalizers: list[float],
    inner_folds: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    spec, oof, inner_cindex = select_spec_and_oof(
        values[outer_train],
        survival[outer_train],
        events[outer_train],
        kind,
        split_seed,
        pca_dims,
        penalizers,
        inner_folds,
    )
    train_risk, test_risk = fit_predict(
        values,
        survival,
        events,
        outer_train,
        heldout,
        kind,
        spec,
        (split_seed * 1000 + 7) % (2**32 - 1),
    )
    heldout_percentile = empirical_percentile(train_risk, test_risk)
    metadata = {
        "pca_dim": spec.pca_dim,
        "penalizer": float(spec.penalizer),
        "inner_cindex": float(inner_cindex),
    }
    return oof, heldout_percentile, metadata


def select_bimodal_weights(
    oof_risks: np.ndarray,
    survival: np.ndarray,
    events: np.ndarray,
    step: float,
) -> tuple[np.ndarray, float]:
    units = int(round(1.0 / step))
    rows = []
    for tabular_units in range(units + 1):
        weights = np.asarray(
            [tabular_units, units - tabular_units], dtype=np.float64
        ) / units
        score = cindex(survival, events, oof_risks @ weights)
        nonzero = int((weights > 0).sum())
        rows.append((score, -nonzero, weights[0], weights))
    rows.sort(key=lambda item: item[:3], reverse=True)
    return rows[0][3], float(rows[0][0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--mean-embeddings", type=Path, required=True)
    parser.add_argument("--moments-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=20260827)
    parser.add_argument("--pca-dims", nargs="+", type=int, default=[4, 8])
    parser.add_argument(
        "--penalizers", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0]
    )
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    if args.outer_folds < 2 or args.outer_repeats < 1 or args.inner_folds < 2:
        raise ValueError("CV folds/repeats must define a valid nested protocol")
    if args.bootstrap_iterations < 1:
        raise ValueError("bootstrap-iterations must be positive")

    tabular = load_indexed_csv(args.features, {"case_id"}).astype(np.float64)
    targets = load_indexed_csv(
        args.targets, {"case_id", "survival_days", "event"}
    )[["survival_days", "event"]]
    text = load_text_npz(args.text_embeddings)
    mean = load_vision_csv(args.mean_embeddings)
    moments = load_vision_csv(args.moments_embeddings)
    common = sorted(
        set(tabular.index)
        & set(targets.index)
        & set(text.index)
        & set(mean.index)
        & set(moments.index)
    )
    cohort = targets.loc[common].copy()
    cohort["survival_days"] = pd.to_numeric(
        cohort["survival_days"], errors="coerce"
    )
    cohort["event"] = pd.to_numeric(cohort["event"], errors="coerce")
    cohort = cohort.loc[
        cohort["survival_days"].notna()
        & (cohort["survival_days"] > 0)
        & cohort["event"].isin([0, 1])
    ].sort_index()
    ids = cohort.index.tolist()
    survival = cohort["survival_days"].to_numpy(dtype=np.float64)
    events = cohort["event"].to_numpy(dtype=np.int64)
    if len(ids) < args.outer_folds * 2:
        raise ValueError("Too few common cases for requested outer CV")
    if np.min(np.bincount(events, minlength=2)) < args.outer_folds:
        raise ValueError("Each outcome stratum must support every outer fold")

    values = {
        "tabular": tabular.loc[ids].to_numpy(dtype=np.float64),
        "text": text.loc[ids].to_numpy(dtype=np.float64),
        "vision_mean": mean.loc[ids].to_numpy(dtype=np.float64),
        "vision_moments": moments.loc[ids].to_numpy(dtype=np.float64),
    }
    kinds = {
        "tabular": "tabular",
        "text": "text",
        "vision_mean": "vision",
        "vision_moments": "vision",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort.reset_index().to_csv(args.output_dir / "cohort_common.csv", index=False)
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    all_indices = np.arange(len(ids))
    started = time.monotonic()

    for repeat in range(args.outer_repeats):
        splitter = StratifiedKFold(
            n_splits=args.outer_folds,
            shuffle=True,
            random_state=args.random_state + repeat,
        )
        for fold, (outer_train, heldout) in enumerate(
            splitter.split(all_indices, events)
        ):
            split_seed = args.random_state + repeat * 100 + fold
            oof: dict[str, np.ndarray] = {}
            heldout_risk: dict[str, np.ndarray] = {}
            modality_metadata: dict[str, dict[str, Any]] = {}
            for name in ("tabular", "text", "vision_mean", "vision_moments"):
                oof[name], heldout_risk[name], modality_metadata[name] = (
                    fit_outer_modality(
                        values[name],
                        survival,
                        events,
                        outer_train,
                        heldout,
                        kinds[name],
                        split_seed,
                        args.pca_dims,
                        args.penalizers,
                        args.inner_folds,
                    )
                )

            mean_train = np.column_stack(
                [oof["tabular"], oof["text"], oof["vision_mean"]]
            )
            moments_train = np.column_stack(
                [oof["tabular"], oof["text"], oof["vision_moments"]]
            )
            mean_test = np.column_stack(
                [
                    heldout_risk["tabular"],
                    heldout_risk["text"],
                    heldout_risk["vision_mean"],
                ]
            )
            moments_test = np.column_stack(
                [
                    heldout_risk["tabular"],
                    heldout_risk["text"],
                    heldout_risk["vision_moments"],
                ]
            )
            bimodal_train = np.column_stack(
                [oof["tabular"], oof["vision_moments"]]
            )
            bimodal_test = np.column_stack(
                [heldout_risk["tabular"], heldout_risk["vision_moments"]]
            )
            mean_weights, mean_fusion_inner = select_fusion_weights(
                mean_train,
                survival[outer_train],
                events[outer_train],
                args.weight_step,
            )
            moments_weights, moments_fusion_inner = select_fusion_weights(
                moments_train,
                survival[outer_train],
                events[outer_train],
                args.weight_step,
            )
            bimodal_weights, bimodal_fusion_inner = select_bimodal_weights(
                bimodal_train,
                survival[outer_train],
                events[outer_train],
                args.weight_step,
            )
            risks = {
                "tabular": heldout_risk["tabular"],
                "text": heldout_risk["text"],
                "vision_mean": heldout_risk["vision_mean"],
                "vision_moments": heldout_risk["vision_moments"],
                "fusion_mean": mean_test @ mean_weights,
                "fusion_moments": moments_test @ moments_weights,
                "fusion_bimodal_moments": bimodal_test @ bimodal_weights,
            }
            scores = {
                name: cindex(survival[heldout], events[heldout], risk)
                for name, risk in risks.items()
            }
            row: dict[str, Any] = {
                "repeat": repeat,
                "fold": fold,
                "split_seed": split_seed,
                "n_train": len(outer_train),
                "n_heldout": len(heldout),
                "events_train": int(events[outer_train].sum()),
                "events_heldout": int(events[heldout].sum()),
                **{f"cindex_{name}": score for name, score in scores.items()},
                "delta_vision_moments_minus_mean": (
                    scores["vision_moments"] - scores["vision_mean"]
                ),
                "delta_fusion_moments_minus_mean": (
                    scores["fusion_moments"] - scores["fusion_mean"]
                ),
                "delta_fusion_moments_minus_tabular": (
                    scores["fusion_moments"] - scores["tabular"]
                ),
                "delta_bimodal_minus_trimodal_moments": (
                    scores["fusion_bimodal_moments"] - scores["fusion_moments"]
                ),
                "delta_bimodal_minus_tabular": (
                    scores["fusion_bimodal_moments"] - scores["tabular"]
                ),
                "mean_weight_tabular": float(mean_weights[0]),
                "mean_weight_text": float(mean_weights[1]),
                "mean_weight_vision": float(mean_weights[2]),
                "moments_weight_tabular": float(moments_weights[0]),
                "moments_weight_text": float(moments_weights[1]),
                "moments_weight_vision": float(moments_weights[2]),
                "bimodal_weight_tabular": float(bimodal_weights[0]),
                "bimodal_weight_vision": float(bimodal_weights[1]),
                "mean_fusion_inner_cindex": float(mean_fusion_inner),
                "moments_fusion_inner_cindex": float(moments_fusion_inner),
                "bimodal_fusion_inner_cindex": float(bimodal_fusion_inner),
            }
            for name, metadata in modality_metadata.items():
                row[f"{name}_pca_dim"] = metadata["pca_dim"]
                row[f"{name}_penalizer"] = metadata["penalizer"]
                row[f"{name}_inner_cindex"] = metadata["inner_cindex"]
            fold_rows.append(row)

            heldout_set = set(heldout.tolist())
            split_rows.extend(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "case_id": ids[index],
                    "partition": (
                        "heldout" if index in heldout_set else "outer_train"
                    ),
                    "survival_days": float(survival[index]),
                    "event": int(events[index]),
                }
                for index in all_indices
            )
            prediction_rows.extend(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "case_id": ids[index],
                    "survival_days": float(survival[index]),
                    "event": int(events[index]),
                    **{
                        f"risk_{name}": float(risk[local])
                        for name, risk in risks.items()
                    },
                }
                for local, index in enumerate(heldout)
            )
            print(
                f"repeat={repeat} fold={fold} "
                f"vision(mean/moments)={scores['vision_mean']:.4f}/"
                f"{scores['vision_moments']:.4f} "
                f"fusion(mean/moments)={scores['fusion_mean']:.4f}/"
                f"{scores['fusion_moments']:.4f} "
                f"bimodal={scores['fusion_bimodal_moments']:.4f} "
                f"weights_moments={moments_weights.tolist()} "
                f"weights_bimodal={bimodal_weights.tolist()}",
                flush=True,
            )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    repeats = repeat_metrics(predictions)
    bootstrap = clustered_patient_bootstrap(
        predictions,
        (
            ("vision_moments", "vision_mean"),
            ("fusion_moments", "fusion_mean"),
            ("fusion_moments", "tabular"),
            ("fusion_moments", "vision_moments"),
            ("fusion_bimodal_moments", "fusion_moments"),
            ("fusion_bimodal_moments", "tabular"),
            ("fusion_bimodal_moments", "vision_moments"),
        ),
        args.bootstrap_iterations,
        args.random_state + 50000,
    )
    folds.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    repeats.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_cluster_bootstrap.csv", index=False)
    predictions.to_csv(args.output_dir / "heldout_predictions.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.output_dir / "splits.csv", index=False)

    results = {}
    for name in MODEL_NAMES:
        column = f"cindex_{name}"
        results[name] = {
            "mean_pooled_repeat_cindex": float(repeats[column].mean()),
            "std_across_repeats": float(repeats[column].std(ddof=1)),
            "per_repeat": repeats[column].astype(float).tolist(),
            "mean_outer_fold_cindex": float(folds[column].mean()),
        }
    fold_stability = {}
    for column in (
        "delta_vision_moments_minus_mean",
        "delta_fusion_moments_minus_mean",
        "delta_fusion_moments_minus_tabular",
        "delta_bimodal_minus_trimodal_moments",
        "delta_bimodal_minus_tabular",
    ):
        values_column = folds[column]
        fold_stability[column] = {
            "mean": float(values_column.mean()),
            "median": float(values_column.median()),
            "positive_folds": int((values_column > 0).sum()),
            "ties": int((values_column == 0).sum()),
            "negative_folds": int((values_column < 0).sum()),
        }
    weight_summary = {}
    for variant in ("mean", "moments"):
        weight_summary[variant] = {}
        for modality in ("tabular", "text", "vision"):
            column = f"{variant}_weight_{modality}"
            weight_summary[variant][modality] = {
                "mean": float(folds[column].mean()),
                "std": float(folds[column].std(ddof=1)),
                "zero_fraction": float((folds[column] == 0).mean()),
            }

    weight_summary["bimodal"] = {
        modality: {
            "mean": float(folds[f"bimodal_weight_{modality}"].mean()),
            "std": float(folds[f"bimodal_weight_{modality}"].std(ddof=1)),
            "zero_fraction": float(
                (folds[f"bimodal_weight_{modality}"] == 0).mean()
            ),
        }
        for modality in ("tabular", "vision")
    }

    summary = {
        "status": "stunet_trimodal_pooling_nested_repeated_cv",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.monotonic() - started),
        "cohort": {"n_cases": len(ids), "n_events": int(events.sum())},
        "protocol": {
            "outer_folds": args.outer_folds,
            "outer_repeats": args.outer_repeats,
            "inner_folds": args.inner_folds,
            "random_state": args.random_state,
            "pca_dims": args.pca_dims,
            "penalizers": args.penalizers,
            "fusion": "train-only empirical-risk percentiles and simplex weights",
            "weight_step": args.weight_step,
            "patient_clustered_bootstrap_iterations": args.bootstrap_iterations,
        },
        "results": results,
        "paired_comparisons": bootstrap.to_dict(orient="records"),
        "fold_stability": fold_stability,
        "weight_stability": weight_summary,
        "claim_boundary": (
            "Internal repeated nested CV on a previously explored cohort; external "
            "validation remains required."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    provenance = {
        "script": str(Path(__file__).resolve()),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in {
                "features": args.features,
                "targets": args.targets,
                "text_embeddings": args.text_embeddings,
                "mean_embeddings": args.mean_embeddings,
                "moments_embeddings": args.moments_embeddings,
            }.items()
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "publication_boundary": (
            "Patient-level cohort, splits and predictions remain local; only aggregate "
            "fold/repeat/bootstrap metrics and non-identifying provenance are tracked."
        ),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"results": results, "comparisons": summary["paired_comparisons"]},
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
