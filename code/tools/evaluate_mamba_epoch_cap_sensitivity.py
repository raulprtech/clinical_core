"""Evaluate conservative Mamba epoch caps on fixed outer-CV folds.

This is a post-hoc robustness analysis. It reuses the exact repeated outer
splits and initialization seeds from the trimodal nested-CV run, replacing the
inner-selected epoch count with min(selected_epochs, cap). It evaluates only
the visual Mamba risk; fusion weights are not refitted.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from evaluate_trimodal_fusion import (  # noqa: E402
    cindex,
    load_indexed_csv,
    load_text_npz,
    load_vision_csv,
)
from evaluate_trimodal_sequence_fusion import fit_sequence_train_test_risk  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--vision-embeddings", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--reference-fold-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--caps", nargs="+", type=int, default=[5, 10, 15])
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=3031)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--mamba-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if any(cap < 1 for cap in args.caps):
        raise ValueError("Epoch caps must be positive")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    tabular = load_indexed_csv(args.features, {"case_id"})
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
    features, positions, mask = pad_sequences(sequences, ids)

    reference = pd.read_csv(args.reference_fold_metrics).set_index(["repeat", "fold"])
    splitter = RepeatedStratifiedKFold(
        n_splits=args.outer_folds,
        n_repeats=args.outer_repeats,
        random_state=args.random_state,
    )
    rows = []
    for split_number, (outer_train, heldout) in enumerate(
        splitter.split(np.zeros(len(events)), events)
    ):
        repeat = split_number // args.outer_folds
        fold = split_number % args.outer_folds
        selected = int(reference.loc[(repeat, fold), "mamba_heldout_epochs"])
        split_seed = args.random_state + repeat * 100 + fold
        original_cindex = float(reference.loc[(repeat, fold), "cindex_vision_mamba"])
        for cap in args.caps:
            used = min(selected, int(cap))
            _, risk = fit_sequence_train_test_risk(
                features,
                positions,
                mask,
                survival,
                events,
                outer_train,
                heldout,
                used,
                split_seed * 10000 + 99,
                args,
                device,
            )
            score = cindex(survival[heldout], events[heldout], risk)
            rows.append({
                "repeat": repeat,
                "fold": fold,
                "cap": int(cap),
                "selected_epochs": selected,
                "epochs_used": used,
                "cindex_original_selected": original_cindex,
                "cindex_capped": score,
                "delta_capped_minus_original": score - original_cindex,
            })
        print(f"repeat={repeat} fold={fold} selected={selected}", flush=True)

    metrics = pd.DataFrame(rows)
    aggregate = (
        metrics.groupby("cap")
        .agg(
            mean_cindex=("cindex_capped", "mean"),
            std_fold_cindex=("cindex_capped", "std"),
            mean_delta=("delta_capped_minus_original", "mean"),
            wins=("delta_capped_minus_original", lambda x: int((x > 0).sum())),
            ties=("delta_capped_minus_original", lambda x: int((x == 0).sum())),
            losses=("delta_capped_minus_original", lambda x: int((x < 0).sum())),
            capped_folds=("selected_epochs", lambda x: 0),
        )
        .reset_index()
    )
    for index, cap in enumerate(aggregate["cap"]):
        aggregate.loc[index, "capped_folds"] = int(
            (metrics.loc[metrics["cap"] == cap, "selected_epochs"] > cap).sum()
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    summary = {
        "status": "posthoc_epoch_cap_sensitivity",
        "caps": [int(cap) for cap in args.caps],
        "reference_mean_fold_cindex": float(
            reference["cindex_vision_mamba"].mean()
        ),
        "aggregate": aggregate.to_dict(orient="records"),
        "claim_boundary": (
            "Post-hoc sensitivity on an explored cohort; not a new confirmatory estimate."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
