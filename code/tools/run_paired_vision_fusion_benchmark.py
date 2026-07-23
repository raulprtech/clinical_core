"""Run a fair ResNet-vs-STU-Net comparison across three trimodal fusions.

This is an orchestration layer.  The statistical and neural implementations
remain in ``evaluate_trimodal_fusion.py`` and
``evaluate_cross_attention_fusion.py``.  Before launching them, this script
restricts both visual encoders to the exact same patient cohort so their
80/20 splits are paired seed by seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_trimodal_fusion import (
    DEFAULT_SEEDS,
    load_indexed_csv,
    load_text_npz,
    load_vision_csv,
)


ENCODERS = ("resnet18_2p5d", "stunet_s_frozen")
FUSIONS = ("concat", "convex", "cross_attention")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_targets(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    aliases = {
        "os_time": "survival_days",
        "os_event": "event",
        "submitter_id": "case_id",
        "patient_id": "case_id",
    }
    for source, target in aliases.items():
        if target not in frame and source in frame:
            frame = frame.rename(columns={source: target})
    required = {"case_id", "survival_days", "event"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing target columns {sorted(missing)}")
    frame["case_id"] = frame["case_id"].astype(str).str.strip().str.upper()
    if frame["case_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate case IDs")
    return frame


def write_vision(frame: pd.DataFrame, path: Path) -> None:
    exported = frame.copy()
    exported.columns = [f"f{index:04d}" for index in range(exported.shape[1])]
    exported.index.name = "case_id"
    exported.reset_index().to_csv(path, index=False)


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def completed(directory: Path) -> bool:
    return (directory / "summary.json").is_file() and (
        directory / "per_seed_metrics.csv"
    ).is_file()


def assert_identical_splits(output_root: Path) -> None:
    split_frames = {}
    for encoder in ENCODERS:
        path = output_root / encoder / "convex" / "splits.csv"
        frame = pd.read_csv(path).sort_values(["seed", "case_id"]).reset_index(drop=True)
        split_frames[encoder] = frame[["seed", "case_id", "partition"]]
    if not split_frames[ENCODERS[0]].equals(split_frames[ENCODERS[1]]):
        raise RuntimeError("Encoder split assignments differ; paired comparison aborted")


def summarize(output_root: Path, n_cases: int, n_events: int, seeds: list[int]) -> None:
    seed_tables = {}
    result_rows = []
    comparison_rows = []

    for encoder in ENCODERS:
        convex = pd.read_csv(output_root / encoder / "convex" / "per_seed_metrics.csv")
        neural = pd.read_csv(
            output_root / encoder / "three_fusion" / "per_seed_metrics.csv"
        )
        merged = convex.merge(
            neural[["seed", "cindex_concat", "cindex_convex", "cindex_cross_attention"]],
            on="seed",
            validate="one_to_one",
        )
        merged.insert(0, "encoder", encoder)
        seed_tables[encoder] = merged
        best_unimodal = merged[
            ["cindex_tabular", "cindex_text", "cindex_vision"]
        ].max(axis=1)
        for fusion in FUSIONS:
            values = merged[f"cindex_{fusion}"]
            delta_vision = values - merged["cindex_vision"]
            delta_best_single = values - best_unimodal
            result_rows.append({
                "encoder": encoder,
                "fusion": fusion,
                "n_cases": n_cases,
                "cindex_mean": float(values.mean()),
                "cindex_std_across_seeds": float(values.std(ddof=1)),
                "delta_vs_vision_mean": float(delta_vision.mean()),
                "wins_vs_vision": int((delta_vision > 0).sum()),
                "delta_vs_best_unimodal_per_seed_mean": float(delta_best_single.mean()),
                "wins_vs_best_unimodal_per_seed": int((delta_best_single > 0).sum()),
            })

    paired = seed_tables[ENCODERS[0]][["seed"]].copy()
    for fusion in FUSIONS:
        resnet = seed_tables[ENCODERS[0]][f"cindex_{fusion}"].to_numpy()
        stunet = seed_tables[ENCODERS[1]][f"cindex_{fusion}"].to_numpy()
        delta = stunet - resnet
        paired[f"delta_stunet_minus_resnet_{fusion}"] = delta
        comparison_rows.append({
            "fusion": fusion,
            "delta_stunet_minus_resnet_mean": float(delta.mean()),
            "delta_std_across_seeds": float(delta.std(ddof=1)),
            "stunet_wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "resnet_wins": int((delta < 0).sum()),
        })

    results = pd.DataFrame(result_rows)
    comparisons = pd.DataFrame(comparison_rows)
    all_seeds = pd.concat(seed_tables.values(), ignore_index=True)
    results.to_csv(output_root / "fusion_results_summary.csv", index=False)
    comparisons.to_csv(output_root / "encoder_paired_summary.csv", index=False)
    paired.to_csv(output_root / "encoder_paired_deltas_per_seed.csv", index=False)
    all_seeds.to_csv(output_root / "all_metrics_per_seed.csv", index=False)

    best = results.sort_values("cindex_mean", ascending=False).iloc[0]
    summary = {
        "status": "paired_resnet_stunet_three_fusion_benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {"n_cases": n_cases, "n_events": n_events},
        "seeds": seeds,
        "paired_design": {
            "same_patient_intersection": True,
            "same_event_stratified_80_20_splits": True,
            "train_only_preprocessing_and_model_selection": True,
        },
        "best_mean_configuration": {
            "encoder": str(best["encoder"]),
            "fusion": str(best["fusion"]),
            "cindex_mean": float(best["cindex_mean"]),
            "cindex_std_across_seeds": float(best["cindex_std_across_seeds"]),
        },
        "result_files": {
            "summary": "fusion_results_summary.csv",
            "encoder_deltas": "encoder_paired_summary.csv",
            "per_seed": "all_metrics_per_seed.csv",
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== RESULTADOS DE FUSION ===")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== STU-NET MENOS RESNET (PAREADO) ===")
    print(comparisons.to_string(index=False, float_format=lambda value: f"{value:+.4f}"))
    print(f"\nMejor media: {best['encoder']} + {best['fusion']} = {best['cindex_mean']:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--resnet-embeddings", type=Path, required=True)
    parser.add_argument("--stunet-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--resume", action="store_true", help="Skip a completed encoder stage"
    )
    args = parser.parse_args()

    for path in (
        args.features,
        args.targets,
        args.text_embeddings,
        args.resnet_embeddings,
        args.stunet_embeddings,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "run_inputs_and_protocol.json"
    requested_manifest = {
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in {
                "features": args.features,
                "targets": args.targets,
                "text_embeddings": args.text_embeddings,
                "resnet_embeddings": args.resnet_embeddings,
                "stunet_embeddings": args.stunet_embeddings,
            }.items()
        },
        "protocol": {
            "seeds": [int(seed) for seed in args.seeds],
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
        },
    }
    if args.resume and manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text())
        if previous_manifest != requested_manifest:
            raise RuntimeError(
                "--resume cannot reuse this output directory because an input or "
                "protocol parameter changed. Choose a new --output-dir."
            )
    elif args.resume and any(args.output_dir.glob("*/convex/summary.json")):
        raise RuntimeError(
            "--resume found legacy results without an input manifest. Choose a new "
            "--output-dir to avoid mixing runs."
        )
    manifest_path.write_text(json.dumps(requested_manifest, indent=2))

    inputs_dir = args.output_dir / "paired_inputs"
    inputs_dir.mkdir(exist_ok=True)

    features = load_indexed_csv(args.features, {"case_id"})
    targets_raw = normalize_targets(args.targets)
    normalized_targets = inputs_dir / "targets_normalized.csv"
    targets_raw.to_csv(normalized_targets, index=False)
    targets = targets_raw.set_index("case_id")
    text = load_text_npz(args.text_embeddings)
    visions = {
        "resnet18_2p5d": load_vision_csv(args.resnet_embeddings),
        "stunet_s_frozen": load_vision_csv(args.stunet_embeddings),
    }
    common = sorted(
        set(features.index)
        & set(targets.index)
        & set(text.index)
        & set(visions[ENCODERS[0]].index)
        & set(visions[ENCODERS[1]].index)
    )
    outcomes = targets.loc[common, ["survival_days", "event"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = (
        outcomes["survival_days"].notna()
        & (outcomes["survival_days"] > 0)
        & outcomes["event"].isin([0, 1])
    )
    ids = outcomes.index[valid].sort_values().tolist()
    if len(ids) < 50:
        raise ValueError(f"Only {len(ids)} patients are common to every modality")
    n_events = int(outcomes.loc[ids, "event"].sum())
    pd.DataFrame({"case_id": ids}).to_csv(
        args.output_dir / "cohort_exact_intersection.csv", index=False
    )
    print(f"Exact paired cohort: n={len(ids)}, events={n_events}", flush=True)

    filtered_vision = {}
    for encoder, frame in visions.items():
        path = inputs_dir / f"{encoder}_common.csv"
        write_vision(frame.loc[ids], path)
        filtered_vision[encoder] = path

    tools_dir = Path(__file__).resolve().parent
    for encoder in ENCODERS:
        encoder_root = args.output_dir / encoder
        convex_dir = encoder_root / "convex"
        neural_dir = encoder_root / "three_fusion"
        base = [
            "--features", str(args.features),
            "--targets", str(normalized_targets),
            "--text-embeddings", str(args.text_embeddings),
            "--vision-embeddings", str(filtered_vision[encoder]),
            "--vision-label", encoder,
            "--seeds", *[str(seed) for seed in args.seeds],
        ]
        if not (args.resume and completed(convex_dir)):
            run([
                sys.executable,
                str(tools_dir / "evaluate_trimodal_fusion.py"),
                *base,
                "--output-dir", str(convex_dir),
            ])
        else:
            print(f"Reusing completed convex stage: {convex_dir}")
        if not (args.resume and completed(neural_dir)):
            run([
                sys.executable,
                str(tools_dir / "evaluate_cross_attention_fusion.py"),
                *base,
                "--convex-results-dir", str(convex_dir),
                "--output-dir", str(neural_dir),
                "--max-epochs", str(args.max_epochs),
                "--patience", str(args.patience),
            ])
        else:
            print(f"Reusing completed neural stage: {neural_dir}")

    assert_identical_splits(args.output_dir)
    summarize(args.output_dir, len(ids), n_events, [int(seed) for seed in args.seeds])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
