"""Paired nested evaluation of symmetric 2.5D slice-context spans.

The architecture is fixed to the predeclared Mamba-64 configuration without
explicit positions. Only the three frozen ResNet18 sequence caches differ.
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

from evaluate_resnet_sequence_models import load_sequences, load_targets, pad_sequences, safe_cindex
from evaluate_resnet_sequence_nested_cv import (
    clustered_patient_bootstrap,
    file_sha256,
    refit_and_predict,
    select_epochs_inner,
)
from evaluate_sequence_factorial_ablation import load_modality_map

SPAN_NAMES = ("span1", "span2", "span4")
COMPARISONS = (("span2", "span1"), ("span4", "span1"), ("span4", "span2"))


def _validate_cache(directory: Path, expected_span: int) -> None:
    provenance_path = directory / "provenance.json"
    manifest_path = directory / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing sequence manifest: {manifest_path}")
    if expected_span == 1 and not provenance_path.exists():
        return
    if not provenance_path.exists():
        raise FileNotFoundError(f"Missing cache provenance: {provenance_path}")
    payload = json.loads(provenance_path.read_text())
    offsets = payload.get("slice_offsets")
    if offsets is None and expected_span == 1:
        context = str(payload.get("context", ""))
        if "[-1, 0, 1]" in context:
            return
    if offsets != [-expected_span, 0, expected_span]:
        raise ValueError(
            f"{directory} has slice_offsets={offsets}; expected "
            f"{[-expected_span, 0, expected_span]}"
        )


def _pooled_repeat_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repeat, frame in predictions.groupby("repeat", sort=True):
        if frame["case_id"].duplicated().any():
            raise ValueError(f"Repeat {repeat} contains duplicate OOF patients")
        for name in SPAN_NAMES:
            row = {
                "repeat": int(repeat),
                "configuration": name,
                "n_all": len(frame),
                "events_all": int(frame["event"].sum()),
            }
            for subgroup in ("all", "CT", "MR"):
                subset = frame if subgroup == "all" else frame[frame["modality"] == subgroup]
                key = subgroup.lower()
                row[f"n_{key}"] = len(subset)
                row[f"events_{key}"] = int(subset["event"].sum())
                row[f"cindex_{key}"] = safe_cindex(
                    subset["survival_days"].to_numpy(dtype=np.float64),
                    subset[f"risk_{name}"].to_numpy(dtype=np.float64),
                    subset["event"].to_numpy(dtype=np.int64),
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--span1-dir", type=Path, required=True)
    parser.add_argument("--span2-dir", type=Path, required=True)
    parser.add_argument("--span4-dir", type=Path, required=True)
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
    args.use_position = False

    if min(args.outer_folds, args.outer_repeats, args.inner_folds) < 1:
        raise ValueError("CV counts must be positive")
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise ValueError("Outer and inner CV require at least two folds")
    if args.epochs < 1 or args.patience < 1 or args.bootstrap_iterations < 1:
        raise ValueError("Training and bootstrap counts must be positive")

    directories = {
        "span1": args.span1_dir,
        "span2": args.span2_dir,
        "span4": args.span4_dir,
    }
    for name, span in zip(SPAN_NAMES, (1, 2, 4)):
        _validate_cache(directories[name], span)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    sequence_sets = {name: load_sequences(path) for name, path in directories.items()}
    targets = load_targets(args.targets)
    modalities = load_modality_map(args.modality_manifest)
    common = set(targets.index) & set(modalities.index)
    for sequences in sequence_sets.values():
        common &= set(sequences)
    cohort = targets.loc[sorted(common)].copy()
    cohort = cohort.loc[
        cohort["survival_days"].notna()
        & (cohort["survival_days"] > 0)
        & cohort["event"].isin([0, 1])
    ].sort_index()
    ids = cohort.index.tolist()
    survival = cohort["survival_days"].to_numpy(dtype=np.float64)
    events = cohort["event"].to_numpy(dtype=np.int64)
    modality = modalities.loc[ids].to_numpy()
    if np.min(np.bincount(events, minlength=2)) < args.outer_folds:
        raise ValueError("Outcome strata cannot support the requested outer folds")

    tensors = {name: pad_sequences(sequences, ids) for name, sequences in sequence_sets.items()}
    for name, (features, _, _) in tensors.items():
        if features.shape[1] > 64:
            raise ValueError(f"{name} contains more than 64 tokens")

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
        fold_risks = {}
        for name in SPAN_NAMES:
            features, positions, mask = tensors[name]
            selected, inner_epochs, inner_scores = select_epochs_inner(
                "mamba", features, positions, mask, survival, events,
                outer_train, split_seed, args, device,
            )
            risk, parameters = refit_and_predict(
                "mamba", features, positions, mask, survival, events,
                outer_train, heldout, selected, split_seed * 1000 + 29,
                args, device,
            )
            fold_risks[name] = risk
            row = {
                "repeat": repeat,
                "fold": fold,
                "split_seed": split_seed,
                "configuration": name,
                "slice_span": int(name.removeprefix("span")),
                "n_train": len(outer_train),
                "n_heldout": len(heldout),
                "events_heldout": int(events[heldout].sum()),
                "selected_epochs": selected,
                "inner_epochs": json.dumps(inner_epochs),
                "inner_cindex_mean": float(np.mean(inner_scores)),
                "parameters": parameters,
            }
            for subgroup in ("all", "CT", "MR"):
                local_mask = (
                    np.ones(len(heldout), dtype=bool)
                    if subgroup == "all" else modality[heldout] == subgroup
                )
                key = subgroup.lower()
                row[f"n_{key}"] = int(local_mask.sum())
                row[f"events_{key}"] = int(events[heldout][local_mask].sum())
                row[f"cindex_{key}"] = safe_cindex(
                    survival[heldout][local_mask], risk[local_mask],
                    events[heldout][local_mask],
                )
            fold_rows.append(row)

        prediction_rows.extend({
            "repeat": repeat,
            "fold": fold,
            "case_id": ids[index],
            "survival_days": float(survival[index]),
            "event": int(events[index]),
            "modality": modality[index],
            **{f"risk_{name}": float(fold_risks[name][local]) for name in SPAN_NAMES},
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
        scores = " ".join(
            f"{name}={safe_cindex(survival[heldout], fold_risks[name], events[heldout]):.3f}"
            for name in SPAN_NAMES
        )
        print(f"repeat={repeat} fold={fold} {scores}", flush=True)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    repeats = _pooled_repeat_metrics(predictions)
    bootstrap = clustered_patient_bootstrap(
        predictions, COMPARISONS, args.bootstrap_iterations, args.random_state + 50000
    )
    folds.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    repeats.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_cluster_bootstrap.csv", index=False)
    predictions.to_csv(args.output_dir / "heldout_predictions.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.output_dir / "splits.csv", index=False)

    results = {}
    for name in SPAN_NAMES:
        selected = repeats[repeats["configuration"] == name]
        results[name] = {
            subgroup: {
                "mean_pooled_repeat_cindex": float(selected[f"cindex_{subgroup}"].mean()),
                "std_across_repeats": float(selected[f"cindex_{subgroup}"].std(ddof=1)),
                "per_repeat": selected[f"cindex_{subgroup}"].tolist(),
                "n_cases_per_repeat": int(selected[f"n_{subgroup}"].iloc[0]),
                "events_per_repeat": int(selected[f"events_{subgroup}"].iloc[0]),
            }
            for subgroup in ("all", "ct", "mr")
        }
    ranking = sorted(
        ({"configuration": name, "mean_cindex": results[name]["all"]["mean_pooled_repeat_cindex"]}
         for name in SPAN_NAMES),
        key=lambda item: item["mean_cindex"], reverse=True,
    )
    summary = {
        "status": "resnet18_2p5d_context_span_nested_repeated_cv",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.monotonic() - started),
        "cohort": {
            "n_cases": len(ids), "n_events": int(events.sum()),
            "n_ct": int((modality == "CT").sum()), "events_ct": int(events[modality == "CT"].sum()),
            "n_mr": int((modality == "MR").sum()), "events_mr": int(events[modality == "MR"].sum()),
        },
        "protocol": {
            "outer_folds": args.outer_folds, "outer_repeats": args.outer_repeats,
            "inner_folds": args.inner_folds, "outer_refit": True,
            "architecture": "mamba", "tokens": 64, "use_position": False,
            "slice_spans": [1, 2, 4],
            "patient_clustered_bootstrap_iterations": args.bootstrap_iterations,
            "device": str(device),
        },
        "ranking": ranking,
        "results": results,
        "paired_comparisons": bootstrap.to_dict(orient="records"),
        "claim_boundary": "Predeclared internal ablation on a reused cohort; external validation required.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = {
        "script": str(Path(__file__).resolve()),
        "sequence_caches": {
            name: {
                "directory": str(directory.resolve()),
                "manifest_sha256": file_sha256(directory / "manifest.csv"),
                "provenance_sha256": file_sha256(directory / "provenance.json"),
            }
            for name, directory in directories.items()
        },
        "targets": str(args.targets.resolve()),
        "targets_sha256": file_sha256(args.targets),
        "modality_manifest": str(args.modality_manifest.resolve()),
        "modality_manifest_sha256": file_sha256(args.modality_manifest),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key != "use_position"
        },
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"ranking": ranking, "comparisons": summary["paired_comparisons"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
