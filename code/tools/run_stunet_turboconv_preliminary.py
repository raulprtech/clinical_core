"""Low-memory orchestrator for the 25/50 STU-Net quantization kill test."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = REPO_ROOT / "code/tools/evaluate_stunet_turboconv.py"
DEFAULT_OUTPUT = REPO_ROOT / "data/embeddings/vision/stunet_turboconv_preliminary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-calibration", type=int, default=25)
    parser.add_argument("--n-evaluation", type=int, default=50)
    return parser.parse_args()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def evaluator_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(EVALUATOR),
        "--output-root",
        str(args.output_root),
        "--n-calibration",
        str(args.n_calibration),
        "--n-evaluation",
        str(args.n_evaluation),
    ]


def main() -> int:
    args = parse_args()
    split_path = args.output_root / "splits/calibration.csv"
    if not split_path.exists():
        subprocess.run(
            evaluator_command(args) + ["--prepare-only"],
            cwd=REPO_ROOT,
            check=True,
        )
    with split_path.open(newline="") as handle:
        cases = list(csv.DictReader(handle))
    cases.sort(key=lambda row: float(row["estimated_voxels"]), reverse=True)
    shard_root = args.output_root / "calibration/shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for position, row in enumerate(cases, 1):
        case_id = row["case_id"]
        shard = shard_root / f"{case_id}.json"
        if shard.exists():
            print(f"[orchestrator {position}/{len(cases)}] {case_id}: cached", flush=True)
            continue
        print(f"[orchestrator {position}/{len(cases)}] {case_id}: calibrating", flush=True)
        command = evaluator_command(args) + ["--calibration-shard-case", case_id]
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            if error.returncode not in {-9, 137}:
                raise
            print(
                f"[orchestrator {position}/{len(cases)}] {case_id}: "
                "retrying physical-center-patch fallback",
                flush=True,
            )
            subprocess.run(
                command + ["--calibration-low-memory"],
                cwd=REPO_ROOT,
                check=True,
            )

    aggregate: dict[str, dict[str, float]] = {}
    preprocessing_methods: dict[str, str] = {}
    for row in cases:
        payload = json.loads((shard_root / f"{row['case_id']}.json").read_text())
        preprocessing_methods[row["case_id"]] = payload.get(
            "calibration_preprocessing", "nnunet_full_volume"
        )
        for name, values in payload["layers"].items():
            current = aggregate.setdefault(
                name, {"ptq_maxabs": 0.0, "turboconv_maxabs": 0.0}
            )
            for key in current:
                current[key] = max(current[key], float(values[key]))
    atomic_json(
        {
            "schema_version": 1,
            "method": "max_of_independent_patient_shards",
            "n_calibration": len(cases),
            "case_ids": [row["case_id"] for row in cases],
            "rotation_seed": 2026,
            "calibration_preprocessing_by_case": preprocessing_methods,
            "elapsed_seconds": time.perf_counter() - started,
            "layers": aggregate,
        },
        args.output_root / "calibration/activation_maxabs.json",
    )
    evaluation_path = args.output_root / "splits/evaluation.csv"
    with evaluation_path.open(newline="") as handle:
        evaluation = list(csv.DictReader(handle))
    evaluation.sort(key=lambda row: float(row["estimated_voxels"]), reverse=True)
    for variant in ("fp32", "ptq", "turboconv"):
        for position, row in enumerate(evaluation, 1):
            case_id = row["case_id"]
            marker = (
                args.output_root / "variants" / variant / "cases" / case_id
                / "complete.json"
            )
            if marker.exists():
                print(
                    f"[{variant} shard {position}/{len(evaluation)}] {case_id}: cached",
                    flush=True,
                )
                continue
            print(
                f"[{variant} shard {position}/{len(evaluation)}] {case_id}: running",
                flush=True,
            )
            if variant != "fp32":
                subprocess.run(
                    evaluator_command(args) + ["--preprocess-shard-case", case_id],
                    cwd=REPO_ROOT,
                    check=True,
                )
            command = evaluator_command(args) + [
                "--variants", variant,
                "--evaluation-shard-case", case_id,
            ]
            reference_marker = (
                args.output_root / "variants/fp32/cases" / case_id / "complete.json"
            )
            force_low_memory_export = False
            if reference_marker.exists():
                reference = json.loads(reference_marker.read_text())
                force_low_memory_export = (
                    int(reference.get("metrics", {}).get("segmentation_export_order", 1))
                    == 0
                )
            if force_low_memory_export:
                command += ["--export-order", "0"]
            for attempt in range(1, 4):
                attempt_command = (
                    command + ["--export-order", "0"]
                    if attempt == 3 and not force_low_memory_export
                    else command
                )
                try:
                    subprocess.run(attempt_command, cwd=REPO_ROOT, check=True)
                    break
                except subprocess.CalledProcessError as error:
                    if error.returncode not in {-9, 137} or attempt == 3:
                        raise
                    print(
                        f"[{variant} shard {position}/{len(evaluation)}] {case_id}: "
                        f"OOM/SIGKILL retry {attempt}/2",
                        flush=True,
                    )
    return subprocess.run(
        evaluator_command(args) + ["--variants", "fp32", "ptq", "turboconv"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
