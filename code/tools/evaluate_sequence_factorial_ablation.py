"""Factorial ablation of architecture, token count and position features.

Compares attention and Mamba with 32/64 axial tokens and positional features
on/off. All eight configurations share repeated outer folds. Epoch selection
is nested inside each outer-train and every model is reinitialized/refitted on
the complete outer-train. Results are reported globally and separately for CT
and MR, while patient-level predictions remain local.
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

from evaluate_resnet_sequence_models import (  # noqa: E402
    load_sequences,
    load_targets,
    pad_sequences,
    safe_cindex,
)
from evaluate_resnet_sequence_nested_cv import (  # noqa: E402
    clustered_patient_bootstrap,
    file_sha256,
    refit_and_predict,
    select_epochs_inner,
)


ARCHITECTURES = ("attention", "mamba")
TOKEN_COUNTS = (32, 64)
POSITION_FLAGS = (False, True)


def cap_sequence_tokens(
    sequences: dict[str, tuple[np.ndarray, np.ndarray]], max_tokens: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    capped = {}
    for case_id, (features, positions) in sequences.items():
        if len(features) <= max_tokens:
            capped[case_id] = (features, positions)
            continue
        indices = np.linspace(0, len(features) - 1, max_tokens)
        indices = np.rint(indices).astype(int)
        if len(np.unique(indices)) != max_tokens:
            raise RuntimeError("Uniform token selection produced duplicate indices")
        capped[case_id] = (features[indices], positions[indices])
    return capped


def configuration_name(architecture: str, tokens: int, use_position: bool) -> str:
    return f"{architecture}_t{tokens}_pos{'on' if use_position else 'off'}"


def load_modality_map(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    missing = {"case_id", "Modality"} - set(frame.columns)
    if missing:
        raise ValueError(f"Modality manifest is missing {sorted(missing)}")
    frame["case_id"] = frame["case_id"].astype(str).str.strip().str.upper()
    frame["Modality"] = frame["Modality"].astype(str).str.strip().str.upper()
    if frame["case_id"].duplicated().any():
        raise ValueError("Modality manifest contains duplicate case IDs")
    if not frame["Modality"].isin(["CT", "MR"]).all():
        raise ValueError("Only CT and MR modalities are supported")
    return frame.set_index("case_id")["Modality"]


def pooled_repeat_metrics(
    predictions: pd.DataFrame, config_names: list[str]
) -> pd.DataFrame:
    rows = []
    for repeat, frame in predictions.groupby("repeat", sort=True):
        if frame["case_id"].duplicated().any():
            raise ValueError(f"Repeat {repeat} contains duplicate OOF patients")
        for name in config_names:
            row = {
                "repeat": int(repeat),
                "configuration": name,
                "n_all": len(frame),
                "events_all": int(frame["event"].sum()),
            }
            for subgroup in ("all", "CT", "MR"):
                subset = frame if subgroup == "all" else frame[frame["modality"] == subgroup]
                row[f"n_{subgroup.lower()}"] = len(subset)
                row[f"events_{subgroup.lower()}"] = int(subset["event"].sum())
                row[f"cindex_{subgroup.lower()}"] = safe_cindex(
                    subset["survival_days"].to_numpy(dtype=np.float64),
                    subset[f"risk_{name}"].to_numpy(dtype=np.float64),
                    subset["event"].to_numpy(dtype=np.int64),
                )
            rows.append(row)
    return pd.DataFrame(rows)


def comparisons() -> tuple[tuple[str, str], ...]:
    pairs = []
    for tokens in TOKEN_COUNTS:
        for position in POSITION_FLAGS:
            pairs.append((
                configuration_name("mamba", tokens, position),
                configuration_name("attention", tokens, position),
            ))
    for architecture in ARCHITECTURES:
        for position in POSITION_FLAGS:
            pairs.append((
                configuration_name(architecture, 64, position),
                configuration_name(architecture, 32, position),
            ))
    for architecture in ARCHITECTURES:
        for tokens in TOKEN_COUNTS:
            pairs.append((
                configuration_name(architecture, tokens, True),
                configuration_name(architecture, tokens, False),
            ))
    return tuple(pairs)


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
    if np.min(np.bincount(events, minlength=2)) < args.outer_folds:
        raise ValueError("Outcome strata cannot support the requested outer folds")

    tensors = {}
    for token_count in TOKEN_COUNTS:
        tensors[token_count] = pad_sequences(
            cap_sequence_tokens(sequences, token_count), ids
        )
    configs = [
        (architecture, tokens, use_position)
        for architecture in ARCHITECTURES
        for tokens in TOKEN_COUNTS
        for use_position in POSITION_FLAGS
    ]
    names = [
        configuration_name(architecture, tokens, use_position)
        for architecture, tokens, use_position in configs
    ]

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
        for architecture, token_count, use_position in configs:
            name = configuration_name(architecture, token_count, use_position)
            features, positions, mask = tensors[token_count]
            args.use_position = use_position
            selected, inner_epochs, inner_scores = select_epochs_inner(
                architecture,
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
            refit_offset = 17 if architecture == "attention" else 29
            risk, parameters = refit_and_predict(
                architecture,
                features,
                positions,
                mask,
                survival,
                events,
                outer_train,
                heldout,
                selected,
                split_seed * 1000 + refit_offset,
                args,
                device,
            )
            fold_risks[name] = risk
            row = {
                "repeat": repeat,
                "fold": fold,
                "split_seed": split_seed,
                "configuration": name,
                "architecture": architecture,
                "token_count": token_count,
                "use_position": use_position,
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
                    if subgroup == "all"
                    else modality[heldout] == subgroup
                )
                row[f"n_{subgroup.lower()}"] = int(local_mask.sum())
                row[f"events_{subgroup.lower()}"] = int(
                    events[heldout][local_mask].sum()
                )
                row[f"cindex_{subgroup.lower()}"] = safe_cindex(
                    survival[heldout][local_mask],
                    risk[local_mask],
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
            **{
                f"risk_{name}": float(fold_risks[name][local])
                for name in names
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
        fold_summary = " ".join(
            f"{name}={safe_cindex(survival[heldout], fold_risks[name], events[heldout]):.3f}"
            for name in names
        )
        print(f"repeat={repeat} fold={fold} {fold_summary}", flush=True)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    repeats = pooled_repeat_metrics(predictions, names)
    bootstrap = clustered_patient_bootstrap(
        predictions,
        comparisons(),
        args.bootstrap_iterations,
        args.random_state + 50000,
    )
    folds.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    repeats.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_cluster_bootstrap.csv", index=False)
    predictions.to_csv(args.output_dir / "heldout_predictions.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.output_dir / "splits.csv", index=False)

    results = {}
    for name in names:
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
    ranking = sorted(
        (
            {
                "configuration": name,
                "mean_cindex": results[name]["all"]["mean_pooled_repeat_cindex"],
            }
            for name in names
        ),
        key=lambda item: item["mean_cindex"],
        reverse=True,
    )
    summary = {
        "status": "factorial_sequence_ablation_nested_repeated_cv",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.monotonic() - started),
        "cohort": {
            "n_cases": len(ids),
            "n_events": int(events.sum()),
            "n_ct": int((modality == "CT").sum()),
            "n_mr": int((modality == "MR").sum()),
            "events_ct": int(events[modality == "CT"].sum()),
            "events_mr": int(events[modality == "MR"].sum()),
        },
        "protocol": {
            "outer_folds": args.outer_folds,
            "outer_repeats": args.outer_repeats,
            "inner_folds": args.inner_folds,
            "outer_refit": True,
            "architectures": list(ARCHITECTURES),
            "token_counts": list(TOKEN_COUNTS),
            "position_flags": list(POSITION_FLAGS),
            "patient_clustered_bootstrap_iterations": args.bootstrap_iterations,
            "device": str(device),
        },
        "ranking": ranking,
        "results": results,
        "paired_comparisons": bootstrap.to_dict(orient="records"),
        "claim_boundary": (
            "Internal factorial ablation; CT/MR subgroup estimates are exploratory."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
    print(json.dumps({"ranking": ranking, "comparisons": summary["paired_comparisons"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
