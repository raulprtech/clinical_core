"""Leakage-safe fixed ensemble with scaling learned from each outer-train.

Mamba-64 and attention-32 are selected and refitted independently inside each
outer fold. Empirical-percentile and z-score transformations are estimated
only from risks produced for that outer-train and then applied unchanged to
held-out risks. Ensemble weights remain fixed at 0.5/0.5.
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

from evaluate_fixed_sequence_ensemble import per_fold_metrics  # noqa: E402
from evaluate_resnet_sequence_models import (  # noqa: E402
    load_sequences,
    load_targets,
    pad_sequences,
    seed_everything,
)
from evaluate_resnet_sequence_nested_cv import (  # noqa: E402
    clustered_patient_bootstrap,
    file_sha256,
    make_model,
    select_epochs_inner,
    train_epoch,
)
from evaluate_sequence_factorial_ablation import (  # noqa: E402
    cap_sequence_tokens,
    load_modality_map,
    pooled_repeat_metrics,
)


MODEL_SPECS = {
    "attention": {"tokens": 32, "refit_offset": 17},
    "mamba": {"tokens": 64, "refit_offset": 29},
}
OUTPUT_NAMES = (
    "mamba",
    "attention",
    "ensemble_train_percentile50",
    "ensemble_raw50",
    "ensemble_train_z50",
)


def train_ecdf_percentile(
    reference_risk: np.ndarray, heldout_risk: np.ndarray
) -> np.ndarray:
    reference = np.asarray(reference_risk, dtype=np.float64)
    heldout = np.asarray(heldout_risk, dtype=np.float64)
    if reference.ndim != 1 or heldout.ndim != 1 or len(reference) < 2:
        raise ValueError("ECDF inputs must be one-dimensional with >=2 references")
    if not np.isfinite(reference).all() or not np.isfinite(heldout).all():
        raise ValueError("ECDF inputs must be finite")
    ordered = np.sort(reference)
    return np.searchsorted(ordered, heldout, side="right") / len(ordered)


def train_zscore(
    reference_risk: np.ndarray, heldout_risk: np.ndarray
) -> tuple[np.ndarray, float, float]:
    reference = np.asarray(reference_risk, dtype=np.float64)
    heldout = np.asarray(heldout_risk, dtype=np.float64)
    mean = float(reference.mean())
    std = float(reference.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Train risk standard deviation must be positive")
    return (heldout - mean) / std, mean, std


def refit_predict_train_and_heldout(
    name: str,
    features: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    survival: np.ndarray,
    events: np.ndarray,
    outer_train: np.ndarray,
    heldout: np.ndarray,
    epochs: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int]:
    seed_everything(seed)
    model = make_model(name, args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_features = features[outer_train].to(device)
    train_positions = positions[outer_train].to(device)
    train_mask = mask[outer_train].to(device)
    train_survival = torch.tensor(
        survival[outer_train], dtype=torch.float32, device=device
    )
    train_events = torch.tensor(
        events[outer_train], dtype=torch.float32, device=device
    )
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
        heldout_risk = model(
            features[heldout].to(device),
            positions[heldout].to(device),
            mask[heldout].to(device),
        ).cpu().numpy().astype(np.float64)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return train_risk, heldout_risk, int(parameters)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--modality-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=4049)
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
    args.use_position = False

    sequences = load_sequences(args.sequence_dir)
    targets = load_targets(args.targets)
    modalities = load_modality_map(args.modality_manifest)
    common = sorted(set(sequences) & set(targets.index) & set(modalities.index))
    cohort = targets.loc[common].copy()
    cohort = cohort.loc[
        cohort["survival_days"].notna()
        & (cohort["survival_days"] > 0)
        & cohort["event"].isin([0, 1])
    ].sort_index()
    ids = cohort.index.tolist()
    survival = cohort["survival_days"].to_numpy(dtype=np.float64)
    events = cohort["event"].to_numpy(dtype=np.int64)
    modality = modalities.loc[ids].to_numpy()
    tensors = {
        tokens: pad_sequences(cap_sequence_tokens(sequences, tokens), ids)
        for tokens in (32, 64)
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "case_id": ids,
        "survival_days": survival,
        "event": events,
        "modality": modality,
    }).to_csv(args.output_dir / "cohort_common.csv", index=False)
    splitter = RepeatedStratifiedKFold(
        n_splits=args.outer_folds,
        n_repeats=args.outer_repeats,
        random_state=args.random_state,
    )
    training_rows = []
    prediction_rows = []
    split_rows = []
    started = time.monotonic()
    for split_number, (outer_train, heldout) in enumerate(
        splitter.split(np.zeros(len(events)), events)
    ):
        repeat = split_number // args.outer_folds
        fold = split_number % args.outer_folds
        split_seed = args.random_state + repeat * 100 + fold
        train_risks = {}
        heldout_risks = {}
        for name, spec in MODEL_SPECS.items():
            features, positions, mask = tensors[spec["tokens"]]
            selected, inner_epochs, inner_scores = select_epochs_inner(
                name,
                features,
                positions,
                mask,
                survival,
                events,
                outer_train,
                split_seed,
                args,
                device,
            )
            train_risk, heldout_risk, parameters = refit_predict_train_and_heldout(
                name,
                features,
                positions,
                mask,
                survival,
                events,
                outer_train,
                heldout,
                selected,
                split_seed * 1000 + spec["refit_offset"],
                args,
                device,
            )
            train_risks[name] = train_risk
            heldout_risks[name] = heldout_risk
            _, train_mean, train_std = train_zscore(train_risk, heldout_risk)
            training_rows.append({
                "repeat": repeat,
                "fold": fold,
                "model": name,
                "tokens": spec["tokens"],
                "use_position": False,
                "n_outer_train": len(outer_train),
                "n_heldout": len(heldout),
                "selected_epochs": selected,
                "inner_epochs": json.dumps(inner_epochs),
                "inner_cindex_mean": float(np.mean(inner_scores)),
                "train_risk_mean": train_mean,
                "train_risk_std": train_std,
                "parameters": parameters,
            })

        percentile = {
            name: train_ecdf_percentile(train_risks[name], heldout_risks[name])
            for name in MODEL_SPECS
        }
        zscore = {
            name: train_zscore(train_risks[name], heldout_risks[name])[0]
            for name in MODEL_SPECS
        }
        risks = {
            "mamba": heldout_risks["mamba"],
            "attention": heldout_risks["attention"],
            "ensemble_train_percentile50": 0.5
            * (percentile["mamba"] + percentile["attention"]),
            "ensemble_raw50": 0.5
            * (heldout_risks["mamba"] + heldout_risks["attention"]),
            "ensemble_train_z50": 0.5 * (zscore["mamba"] + zscore["attention"]),
        }
        prediction_rows.extend({
            "repeat": repeat,
            "fold": fold,
            "case_id": ids[index],
            "survival_days": float(survival[index]),
            "event": int(events[index]),
            "modality": modality[index],
            **{
                f"risk_{name}": float(risks[name][local])
                for name in OUTPUT_NAMES
            },
        } for local, index in enumerate(heldout))
        heldout_set = set(heldout.tolist())
        split_rows.extend({
            "repeat": repeat,
            "fold": fold,
            "case_id": ids[index],
            "partition": "heldout" if index in heldout_set else "outer_train",
            "event": int(events[index]),
            "modality": modality[index],
        } for index in range(len(ids)))
        print(f"repeat={repeat} fold={fold} complete", flush=True)

    predictions = pd.DataFrame(prediction_rows)
    folds = per_fold_metrics(predictions, OUTPUT_NAMES)
    repeats = pooled_repeat_metrics(predictions, list(OUTPUT_NAMES))
    comparisons = (
        ("ensemble_train_percentile50", "mamba"),
        ("ensemble_train_percentile50", "attention"),
        ("ensemble_raw50", "mamba"),
        ("ensemble_train_z50", "mamba"),
    )
    bootstrap = clustered_patient_bootstrap(
        predictions,
        comparisons,
        args.bootstrap_iterations,
        args.random_state + 70000,
    )
    results = {}
    for name in OUTPUT_NAMES:
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
        "status": "outer_train_scaled_fixed_sequence_ensemble",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.monotonic() - started),
        "cohort": {
            "n_cases": len(ids),
            "n_events": int(events.sum()),
            "n_ct": int((modality == "CT").sum()),
            "n_mr": int((modality == "MR").sum()),
        },
        "protocol": {
            "outer_folds": args.outer_folds,
            "outer_repeats": args.outer_repeats,
            "inner_folds": args.inner_folds,
            "weights": {"mamba": 0.5, "attention": 0.5},
            "primary_scaling": "outer_train_empirical_cdf",
            "sensitivity_scaling": ["raw_risk", "outer_train_zscore"],
            "heldout_used_for_scaling": False,
            "bootstrap_iterations": args.bootstrap_iterations,
        },
        "results": results,
        "paired_comparisons": bootstrap.to_dict(orient="records"),
        "claim_boundary": (
            "Leakage-safe internal follow-up of a post-hoc hypothesis on the "
            "same cohort; not external confirmation."
        ),
    }
    folds.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    repeats.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    pd.DataFrame(training_rows).to_csv(
        args.output_dir / "training_scaling_by_fold.csv", index=False
    )
    bootstrap.to_csv(args.output_dir / "paired_cluster_bootstrap.csv", index=False)
    predictions.to_csv(args.output_dir / "heldout_predictions.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.output_dir / "splits.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    provenance = {
        "script": str(Path(__file__).resolve()),
        "sequence_manifest": str((args.sequence_dir / "manifest.csv").resolve()),
        "sequence_manifest_sha256": file_sha256(args.sequence_dir / "manifest.csv"),
        "targets": str(args.targets.resolve()),
        "targets_sha256": file_sha256(args.targets),
        "modality_manifest": str(args.modality_manifest.resolve()),
        "modality_manifest_sha256": file_sha256(args.modality_manifest),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "use_position"
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
