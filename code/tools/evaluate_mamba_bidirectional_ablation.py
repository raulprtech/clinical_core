"""Paired ablation of unidirectional and shared-weight bidirectional Mamba.

Both models use 64 uniformly sampled ResNet18 axial tokens without explicit
position features. The bidirectional model reuses the exact same Mamba weights
for inferior-to-superior and superior-to-inferior scans. Repeated outer folds,
inner epoch selection and full outer-train refits are paired. Patient-level
artifacts remain local.
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
from evaluate_sequence_factorial_ablation import (  # noqa: E402
    cap_sequence_tokens,
    load_modality_map,
    pooled_repeat_metrics,
)


CONFIGURATIONS = (
    ("mamba_unidirectional", "mamba"),
    ("mamba_bidirectional", "mamba_bidirectional"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--modality-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
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
    if np.min(np.bincount(events, minlength=2)) < args.outer_folds:
        raise ValueError("Outcome strata cannot support the requested outer folds")

    features, positions, mask = pad_sequences(
        cap_sequence_tokens(sequences, args.max_tokens), ids
    )
    names = [configuration for configuration, _ in CONFIGURATIONS]
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
        for configuration, model_name in CONFIGURATIONS:
            selected, inner_epochs, inner_scores = select_epochs_inner(
                model_name,
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
            risk, parameters = refit_and_predict(
                model_name,
                features,
                positions,
                mask,
                survival,
                events,
                outer_train,
                heldout,
                selected,
                split_seed * 1000 + 29,
                args,
                device,
            )
            fold_risks[configuration] = risk
            row = {
                "repeat": repeat,
                "fold": fold,
                "split_seed": split_seed,
                "configuration": configuration,
                "model_name": model_name,
                "bidirectional": model_name == "mamba_bidirectional",
                "max_tokens": args.max_tokens,
                "use_position": False,
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
        print(
            f"repeat={repeat} fold={fold} "
            + " ".join(
                f"{name}={safe_cindex(survival[heldout], fold_risks[name], events[heldout]):.3f}"
                for name in names
            ),
            flush=True,
        )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    repeats = pooled_repeat_metrics(predictions, names)
    bootstrap = clustered_patient_bootstrap(
        predictions,
        (("mamba_bidirectional", "mamba_unidirectional"),),
        args.bootstrap_iterations,
        args.random_state + 60000,
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
    parameter_counts = {
        name: int(folds.loc[folds["configuration"] == name, "parameters"].iloc[0])
        for name in names
    }
    summary = {
        "status": "shared_weight_bidirectional_mamba_ablation",
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
            "max_tokens": args.max_tokens,
            "use_position": False,
            "shared_weights": True,
            "patient_clustered_bootstrap_iterations": args.bootstrap_iterations,
            "device": str(device),
        },
        "parameter_counts": parameter_counts,
        "results": results,
        "paired_comparison": bootstrap.iloc[0].to_dict(),
        "claim_boundary": (
            "Internal architecture ablation; CT/MR subgroup estimates are exploratory."
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
    print(json.dumps({
        "results": results,
        "parameter_counts": parameter_counts,
        "comparison": summary["paired_comparison"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
