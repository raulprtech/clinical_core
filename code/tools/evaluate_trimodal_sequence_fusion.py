"""Compare ResNet embedding risk and Mamba sequence risk in convex fusion.

For every paired outer holdout, tabular/text/ResNet risks follow the original
train-only evaluator. Mamba outer-train risks are cross-fitted; epoch selection
for each cross-fit model is nested inside its training subset. A separate Mamba
model selects epochs inside the complete outer-train, refits on that entire
partition and predicts the untouched held-out patients.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split


CODE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for candidate in (CODE_ROOT, TOOLS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evaluate_resnet_sequence_models import (  # noqa: E402
    load_sequences,
    pad_sequences,
    safe_cindex,
    seed_everything,
)
from components.processors.prognosis.utils.statistical_tests import (  # noqa: E402
    paired_cindex_test,
)
from evaluate_resnet_sequence_nested_cv import (  # noqa: E402
    make_model,
    select_epochs_inner,
    train_epoch,
)
from evaluate_trimodal_fusion import (  # noqa: E402
    DEFAULT_SEEDS,
    cindex,
    empirical_percentile,
    load_indexed_csv,
    load_text_npz,
    load_vision_csv,
    select_fusion_weights,
    select_spec_and_oof,
    fit_predict,
)


def fit_sequence_train_test_risk(
    features: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    survival: np.ndarray,
    events: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    seed_everything(seed)
    model = make_model("mamba", args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_features = features[train_idx].to(device)
    train_positions = positions[train_idx].to(device)
    train_mask = mask[train_idx].to(device)
    train_survival = torch.tensor(
        survival[train_idx], dtype=torch.float32, device=device
    )
    train_events = torch.tensor(events[train_idx], dtype=torch.float32, device=device)
    for _ in range(epochs):
        train_epoch(
            model,
            optimizer,
            train_features,
            train_positions,
            train_mask,
            train_survival,
            train_events,
        )
    model.eval()
    with torch.inference_mode():
        train_risk = model(
            train_features, train_positions, train_mask
        ).cpu().numpy().astype(np.float64)
        test_risk = model(
            features[test_idx].to(device),
            positions[test_idx].to(device),
            mask[test_idx].to(device),
        ).cpu().numpy().astype(np.float64)
    return train_risk, test_risk


def crossfit_mamba_risk(
    features: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    survival: np.ndarray,
    events: np.ndarray,
    outer_train: np.ndarray,
    heldout: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    splitter = StratifiedKFold(
        n_splits=args.inner_folds, shuffle=True, random_state=seed
    )
    oof = np.full(len(outer_train), np.nan, dtype=np.float64)
    crossfit_epochs = []
    for fold, (cross_train_local, validation_local) in enumerate(
        splitter.split(outer_train, events[outer_train])
    ):
        cross_train = outer_train[cross_train_local]
        validation = outer_train[validation_local]
        selected, _, _ = select_epochs_inner(
            "mamba",
            features,
            positions,
            mask,
            survival,
            events,
            cross_train,
            seed * 100 + fold,
            args,
            device,
        )
        train_risk, validation_risk = fit_sequence_train_test_risk(
            features,
            positions,
            mask,
            survival,
            events,
            cross_train,
            validation,
            selected,
            seed * 10000 + fold,
            args,
            device,
        )
        oof[validation_local] = empirical_percentile(train_risk, validation_risk)
        crossfit_epochs.append(selected)
    if not np.isfinite(oof).all():
        raise RuntimeError("Mamba outer-train cross-fitting left missing risks")

    heldout_epochs, heldout_inner_epochs, heldout_inner_scores = select_epochs_inner(
        "mamba",
        features,
        positions,
        mask,
        survival,
        events,
        outer_train,
        seed * 1000 + 77,
        args,
        device,
    )
    outer_train_risk, heldout_risk = fit_sequence_train_test_risk(
        features,
        positions,
        mask,
        survival,
        events,
        outer_train,
        heldout,
        heldout_epochs,
        seed * 10000 + 99,
        args,
        device,
    )
    heldout_percentile = empirical_percentile(outer_train_risk, heldout_risk)
    return oof, heldout_percentile, {
        "crossfit_epochs": crossfit_epochs,
        "heldout_epochs": heldout_epochs,
        "heldout_inner_epochs": heldout_inner_epochs,
        "heldout_inner_cindex_mean": float(np.mean(heldout_inner_scores)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--vision-embeddings", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--inner-folds", type=int, default=3)
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
    if len(ids) < 50:
        raise ValueError(f"Only {len(ids)} common patients have valid outcomes")
    survival = cohort["survival_days"].to_numpy(dtype=np.float64)
    events = cohort["event"].to_numpy(dtype=np.int64)
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
    all_idx = np.arange(len(ids))
    rows = []
    predictions = []
    splits = []
    for seed in args.seeds:
        outer_train, heldout = train_test_split(
            all_idx, test_size=0.20, stratify=events, random_state=int(seed)
        )
        heldout_set = set(heldout.tolist())
        splits.extend({
            "seed": int(seed),
            "case_id": ids[index],
            "partition": "heldout" if index in heldout_set else "outer_train",
            "survival_days": float(survival[index]),
            "event": int(events[index]),
        } for index in all_idx)

        oof = {}
        heldout_risk = {}
        specs = {}
        for modality in ("tabular", "text", "vision"):
            spec, modality_oof, inner_score = select_spec_and_oof(
                values[modality][outer_train],
                survival[outer_train],
                events[outer_train],
                modality,
                int(seed),
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
                int(seed) * 1000 + 7,
            )
            oof[modality] = modality_oof
            heldout_risk[modality] = empirical_percentile(train_risk, test_risk)
            specs[modality] = (spec, inner_score)

        mamba_oof, mamba_heldout, mamba_meta = crossfit_mamba_risk(
            sequence_features,
            sequence_positions,
            sequence_mask,
            survival,
            events,
            outer_train,
            heldout,
            int(seed),
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
        rows.append({
            "seed": int(seed),
            "n_train": len(outer_train),
            "n_heldout": len(heldout),
            "events_heldout": int(events[heldout].sum()),
            **{f"cindex_{name}": value for name, value in scores.items()},
            "delta_fusion_mamba_minus_resnet": (
                scores["fusion_mamba"] - scores["fusion_resnet"]
            ),
            "delta_fusion_mamba_minus_tabular": (
                scores["fusion_mamba"] - scores["tabular"]
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
            "mamba_heldout_inner_cindex_mean": mamba_meta[
                "heldout_inner_cindex_mean"
            ],
        })
        predictions.extend({
            "seed": int(seed),
            "case_id": ids[index],
            "survival_days": float(survival[index]),
            "event": int(events[index]),
            **{f"risk_{name}": float(risk[local]) for name, risk in risks.items()},
        } for local, index in enumerate(heldout))
        print(
            f"seed={seed} tab={scores['tabular']:.4f} "
            f"vision_resnet={scores['vision_resnet']:.4f} "
            f"vision_mamba={scores['vision_mamba']:.4f} "
            f"fusion_resnet={scores['fusion_resnet']:.4f} "
            f"fusion_mamba={scores['fusion_mamba']:.4f} "
            f"weights_mamba={mamba_weights.tolist()}",
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    prediction_frame = pd.DataFrame(predictions)
    bootstrap_rows = []
    comparisons = (
        ("vision_mamba", "vision_resnet"),
        ("fusion_mamba", "fusion_resnet"),
        ("fusion_mamba", "tabular"),
    )
    for seed, frame in prediction_frame.groupby("seed", sort=True):
        for candidate, reference in comparisons:
            paired = paired_cindex_test(
                frame[f"risk_{reference}"].to_numpy(dtype=np.float64),
                frame[f"risk_{candidate}"].to_numpy(dtype=np.float64),
                frame["survival_days"].to_numpy(dtype=np.float64),
                frame["event"].to_numpy(dtype=np.int64),
                n_iter=args.bootstrap_iterations,
                seed=int(seed) + 70000,
            )
            bootstrap_rows.append({
                "seed": int(seed),
                "candidate": candidate,
                "reference": reference,
                **paired,
            })
    bootstrap = pd.DataFrame(bootstrap_rows)
    metrics.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    prediction_frame.to_csv(
        args.output_dir / "heldout_predictions.csv", index=False
    )
    bootstrap.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    pd.DataFrame(splits).to_csv(args.output_dir / "splits.csv", index=False)
    model_names = (
        "tabular",
        "vision_resnet",
        "vision_mamba",
        "fusion_resnet",
        "fusion_mamba",
    )
    results = {
        name: {
            "cindex_mean": float(metrics[f"cindex_{name}"].mean()),
            "cindex_std_across_seeds": float(
                metrics[f"cindex_{name}"].std(ddof=1)
            ),
            "cindex_per_seed": metrics[f"cindex_{name}"].tolist(),
        }
        for name in model_names
    }
    for label, column in (
        ("fusion_mamba_minus_resnet", "delta_fusion_mamba_minus_resnet"),
        ("fusion_mamba_minus_tabular", "delta_fusion_mamba_minus_tabular"),
    ):
        delta = metrics[column]
        results[label] = {
            "mean": float(delta.mean()),
            "std_across_seeds": float(delta.std(ddof=1)),
            "wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "losses": int((delta < 0).sum()),
            "per_seed": delta.tolist(),
        }
    summary = {
        "status": "diagnostic_paired_trimodal_sequence_fusion",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {"n_cases": len(ids), "n_events": int(events.sum())},
        "protocol": {
            "outer_split": "80/20 event-stratified holdout",
            "seeds": [int(seed) for seed in args.seeds],
            "inner_folds": args.inner_folds,
            "mamba_outer_train_risks": "cross-fitted with nested epoch selection",
            "mamba_heldout_risks": "epoch selection inside outer-train then full refit",
            "fusion": "train-only empirical percentiles and simplex weights",
            "weight_step": args.weight_step,
            "device": str(device),
        },
        "results": results,
        "paired_seed_bootstrap": {
            "iterations_per_seed": args.bootstrap_iterations,
            "individual_intervals_excluding_zero": {
                f"{candidate}_minus_{reference}": int(
                    (
                        (bootstrap.loc[
                            (bootstrap["candidate"] == candidate)
                            & (bootstrap["reference"] == reference),
                            "delta_lo",
                        ] > 0)
                        | (bootstrap.loc[
                            (bootstrap["candidate"] == candidate)
                            & (bootstrap["reference"] == reference),
                            "delta_hi",
                        ] < 0)
                    ).sum()
                )
                for candidate, reference in comparisons
            },
        },
        "limitations": [
            "Repeated holdouts overlap and are not independent replicates.",
            "This diagnostic reuses a cohort previously explored during architecture development.",
            "External validation remains required.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = {
        "features": str(args.features.resolve()),
        "targets": str(args.targets.resolve()),
        "text_embeddings": str(args.text_embeddings.resolve()),
        "vision_embeddings": str(args.vision_embeddings.resolve()),
        "sequence_manifest": str((args.sequence_dir / "manifest.csv").resolve()),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
