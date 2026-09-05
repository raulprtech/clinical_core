"""Nested repeated-CV comparison of ResNet and Mamba trimodal fusion.

Every patient receives one outer-fold prediction per repeat. Tabular, text and
ResNet use train-only preprocessing and inner model selection. Mamba risks for
fusion-weight selection are themselves cross-fitted within each outer-train;
the outer held-out risk comes from a separately selected and fully refitted
outer-train model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import RepeatedStratifiedKFold


CODE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for candidate in (CODE_ROOT, TOOLS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evaluate_resnet_sequence_models import load_sequences, pad_sequences  # noqa: E402
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
from evaluate_trimodal_sequence_fusion import crossfit_mamba_risk  # noqa: E402


MODEL_NAMES = (
    "tabular",
    "vision_resnet",
    "vision_mamba",
    "fusion_resnet",
    "fusion_mamba",
)


def repeat_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repeat, frame in predictions.groupby("repeat", sort=True):
        if frame["case_id"].duplicated().any():
            raise ValueError(f"Repeat {repeat} contains duplicate OOF patients")
        row = {
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
        row["delta_fusion_mamba_minus_resnet"] = (
            row["cindex_fusion_mamba"] - row["cindex_fusion_resnet"]
        )
        row["delta_fusion_mamba_minus_tabular"] = (
            row["cindex_fusion_mamba"] - row["cindex_tabular"]
        )
        row["delta_vision_mamba_minus_resnet"] = (
            row["cindex_vision_mamba"] - row["cindex_vision_resnet"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--vision-embeddings", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=3031)
    parser.add_argument("--pca-dims", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--penalizers", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--mamba-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.outer_folds < 2 or args.outer_repeats < 1 or args.inner_folds < 2:
        raise ValueError("CV folds/repeats must define a valid nested protocol")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    tabular = load_indexed_csv(args.features, {"case_id"}).astype(np.float64)
    targets = load_indexed_csv(
        args.targets, {"case_id", "survival_days", "event"}
    )[["survival_days", "event"]]
    text = load_text_npz(args.text_embeddings)
    vision = load_vision_csv(args.vision_embeddings)
    sequences = load_sequences(args.sequence_dir)
    common = sorted(
        set(tabular.index)
        & set(targets.index)
        & set(text.index)
        & set(vision.index)
        & set(sequences)
    )
    cohort = targets.loc[common].copy()
    cohort["survival_days"] = pd.to_numeric(cohort["survival_days"], errors="coerce")
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
        "vision": vision.loc[ids].to_numpy(dtype=np.float64),
    }
    sequence_features, sequence_positions, sequence_mask = pad_sequences(
        sequences, ids
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort.reset_index().to_csv(args.output_dir / "cohort_common.csv", index=False)
    splitter = RepeatedStratifiedKFold(
        n_splits=args.outer_folds,
        n_repeats=args.outer_repeats,
        random_state=args.random_state,
    )
    fold_rows = []
    prediction_rows = []
    split_rows = []
    started = time.monotonic()
    for split_number, (outer_train, heldout) in enumerate(
        splitter.split(np.zeros(len(events)), events)
    ):
        repeat = split_number // args.outer_folds
        fold = split_number % args.outer_folds
        split_seed = args.random_state + repeat * 100 + fold

        oof = {}
        heldout_risk = {}
        for modality in ("tabular", "text", "vision"):
            spec, modality_oof, inner_score = select_spec_and_oof(
                values[modality][outer_train],
                survival[outer_train],
                events[outer_train],
                modality,
                split_seed,
                args.pca_dims,
                args.penalizers,
                args.inner_folds,
            )
            train_risk, test_risk = fit_predict(
                values[modality],
                survival,
                events,
                outer_train,
                heldout,
                modality,
                spec,
                split_seed * 1000 + 7,
            )
            oof[modality] = modality_oof
            heldout_risk[modality] = empirical_percentile(train_risk, test_risk)

        mamba_oof, mamba_heldout, mamba_meta = crossfit_mamba_risk(
            sequence_features,
            sequence_positions,
            sequence_mask,
            survival,
            events,
            outer_train,
            heldout,
            split_seed,
            args,
            device,
        )
        oof["mamba"] = mamba_oof
        heldout_risk["mamba"] = mamba_heldout

        resnet_train = np.column_stack(
            [oof["tabular"], oof["text"], oof["vision"]]
        )
        resnet_test = np.column_stack(
            [heldout_risk["tabular"], heldout_risk["text"], heldout_risk["vision"]]
        )
        mamba_train = np.column_stack(
            [oof["tabular"], oof["text"], oof["mamba"]]
        )
        mamba_test = np.column_stack(
            [heldout_risk["tabular"], heldout_risk["text"], heldout_risk["mamba"]]
        )
        resnet_weights, resnet_inner = select_fusion_weights(
            resnet_train,
            survival[outer_train],
            events[outer_train],
            args.weight_step,
        )
        mamba_weights, mamba_inner = select_fusion_weights(
            mamba_train,
            survival[outer_train],
            events[outer_train],
            args.weight_step,
        )
        risks = {
            "tabular": heldout_risk["tabular"],
            "vision_resnet": heldout_risk["vision"],
            "vision_mamba": heldout_risk["mamba"],
            "fusion_resnet": resnet_test @ resnet_weights,
            "fusion_mamba": mamba_test @ mamba_weights,
        }
        scores = {
            name: cindex(survival[heldout], events[heldout], risk)
            for name, risk in risks.items()
        }
        fold_rows.append({
            "repeat": repeat,
            "fold": fold,
            "split_seed": split_seed,
            "n_train": len(outer_train),
            "n_heldout": len(heldout),
            "events_heldout": int(events[heldout].sum()),
            **{f"cindex_{name}": score for name, score in scores.items()},
            "delta_fusion_mamba_minus_resnet": (
                scores["fusion_mamba"] - scores["fusion_resnet"]
            ),
            "delta_fusion_mamba_minus_tabular": (
                scores["fusion_mamba"] - scores["tabular"]
            ),
            "delta_vision_mamba_minus_resnet": (
                scores["vision_mamba"] - scores["vision_resnet"]
            ),
            "resnet_weight_tabular": float(resnet_weights[0]),
            "resnet_weight_text": float(resnet_weights[1]),
            "resnet_weight_vision": float(resnet_weights[2]),
            "mamba_weight_tabular": float(mamba_weights[0]),
            "mamba_weight_text": float(mamba_weights[1]),
            "mamba_weight_vision": float(mamba_weights[2]),
            "resnet_fusion_inner_cindex": resnet_inner,
            "mamba_fusion_inner_cindex": mamba_inner,
            "mamba_heldout_epochs": mamba_meta["heldout_epochs"],
            "mamba_crossfit_epochs": json.dumps(mamba_meta["crossfit_epochs"]),
        })
        heldout_set = set(heldout.tolist())
        split_rows.extend({
            "repeat": repeat,
            "fold": fold,
            "case_id": ids[index],
            "partition": "heldout" if index in heldout_set else "outer_train",
            "survival_days": float(survival[index]),
            "event": int(events[index]),
        } for index in range(len(ids)))
        prediction_rows.extend({
            "repeat": repeat,
            "fold": fold,
            "case_id": ids[index],
            "survival_days": float(survival[index]),
            "event": int(events[index]),
            **{f"risk_{name}": float(risk[local]) for name, risk in risks.items()},
        } for local, index in enumerate(heldout))
        print(
            f"repeat={repeat} fold={fold} tab={scores['tabular']:.4f} "
            f"vision(resnet/mamba)={scores['vision_resnet']:.4f}/"
            f"{scores['vision_mamba']:.4f} fusion(resnet/mamba)="
            f"{scores['fusion_resnet']:.4f}/{scores['fusion_mamba']:.4f} "
            f"weights_mamba={mamba_weights.tolist()}",
            flush=True,
        )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    repeats = repeat_metrics(predictions)
    bootstrap = clustered_patient_bootstrap(
        predictions,
        (
            ("vision_mamba", "vision_resnet"),
            ("fusion_mamba", "fusion_resnet"),
            ("fusion_mamba", "tabular"),
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
            "std_across_repeats": float(repeats[column].std(ddof=1))
            if len(repeats) > 1
            else None,
            "per_repeat": repeats[column].tolist(),
            "mean_outer_fold_cindex": float(folds[column].mean()),
        }
    weight_summary = {}
    for candidate in ("resnet", "mamba"):
        weight_summary[candidate] = {}
        for modality in ("tabular", "text", "vision"):
            column = f"{candidate}_weight_{modality}"
            weight_summary[candidate][modality] = {
                "mean": float(folds[column].mean()),
                "std": float(folds[column].std(ddof=1)),
                "zero_fraction": float((folds[column] == 0).mean()),
                "values": folds[column].tolist(),
            }
    summary = {
        "status": "confirmatory_trimodal_nested_repeated_cv",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.monotonic() - started),
        "cohort": {"n_cases": len(ids), "n_events": int(events.sum())},
        "protocol": {
            "outer_folds": args.outer_folds,
            "outer_repeats": args.outer_repeats,
            "inner_folds": args.inner_folds,
            "mamba_outer_train_risks": "nested cross-fitted",
            "mamba_heldout_risks": "selected inside and refitted on outer-train",
            "fusion": "train-only risk percentiles and simplex weights",
            "patient_clustered_bootstrap_iterations": args.bootstrap_iterations,
            "device": str(device),
        },
        "results": results,
        "paired_comparisons": bootstrap.to_dict(orient="records"),
        "weight_stability": weight_summary,
        "claim_boundary": (
            "Internal repeated-CV confirmation; external validation remains required."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = {
        "script": str(Path(__file__).resolve()),
        "features": str(args.features.resolve()),
        "features_sha256": file_sha256(args.features),
        "targets": str(args.targets.resolve()),
        "targets_sha256": file_sha256(args.targets),
        "text_embeddings": str(args.text_embeddings.resolve()),
        "text_embeddings_sha256": file_sha256(args.text_embeddings),
        "vision_embeddings": str(args.vision_embeddings.resolve()),
        "vision_embeddings_sha256": file_sha256(args.vision_embeddings),
        "sequence_manifest": str((args.sequence_dir / "manifest.csv").resolve()),
        "sequence_manifest_sha256": file_sha256(args.sequence_dir / "manifest.csv"),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps({"results": results, "comparisons": summary["paired_comparisons"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
