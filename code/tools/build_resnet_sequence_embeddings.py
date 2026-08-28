"""Build a resumable cache of frozen ResNet18 2.5D axial sequences.

Each patient is stored independently under cases/ so interrupted extraction can
resume without rebuilding prior volumes. Outcomes are intentionally absent.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.registry import get_vision_conn  # noqa: E402


def _series_path(row, dicom_dir: Path | None, manifest_dir: Path) -> Path:
    if hasattr(row, "series_dir") and pd.notna(row.series_dir):
        candidate = Path(str(row.series_dir))
        if candidate.exists():
            return candidate.resolve()
        if not candidate.is_absolute():
            candidate = manifest_dir / candidate
        if candidate.exists():
            return candidate.resolve()
    if dicom_dir is None:
        raise FileNotFoundError(
            "Series path is unavailable; pass --dicom-dir or provide series_dir"
        )
    return dicom_dir / str(row.case_id).strip().upper() / str(row.SeriesInstanceUID)


def _valid_case_cache(path: Path, max_tokens: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            features = payload["features"]
            positions = payload["positions"]
        return (
            features.ndim == 2
            and features.shape[1] == 512
            and 1 <= features.shape[0] <= max_tokens
            and positions.shape == (features.shape[0],)
            and np.isfinite(features).all()
            and np.isfinite(positions).all()
        )
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-manifest", required=True, type=Path)
    parser.add_argument("--dicom-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/embeddings/vision/resnet18_2p5d_sequences"),
    )
    parser.add_argument("--weights-dir", type=Path, default=Path("data/models/torch"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--slice-span",
        type=int,
        default=1,
        help=(
            "Symmetric 2.5D channel separation in slice indices; "
            "1 produces [-1, 0, 1], 2 produces [-2, 0, 2], etc."
        ),
    )
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.max_tokens < 2:
        raise ValueError("--max-tokens must be at least 2")
    if args.slice_span < 1:
        raise ValueError("--slice-span must be positive")
    slice_offsets = [-args.slice_span, 0, args.slice_span]
    selected = pd.read_csv(args.series_manifest)
    required = {"case_id", "SeriesInstanceUID"}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"Series manifest is missing {sorted(missing)}")
    selected["case_id"] = selected["case_id"].astype(str).str.strip().str.upper()
    if selected["case_id"].duplicated().any():
        raise ValueError("Series manifest contains duplicate case_id values")

    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    model = get_vision_conn(
        "vision_resnet18_2p5d",
        output_dim=768,
        use_imagenet_weights=True,
        image_size=224,
        window_low=-150,
        window_high=250,
        min_slices=16,
        slice_offsets=slice_offsets,
        device=args.device,
        weights_dir=args.weights_dir.resolve(),
    )
    manifest_dir = args.series_manifest.resolve().parent
    rows = []
    failures = []
    storage_dtype = np.float16 if args.storage_dtype == "float16" else np.float32
    for position, row in enumerate(selected.itertuples(index=False), 1):
        case_id = str(row.case_id)
        output_path = cases_dir / f"{case_id}.npz"
        if not args.force and _valid_case_cache(output_path, args.max_tokens):
            with np.load(output_path, allow_pickle=False) as cached:
                token_count = int(cached["features"].shape[0])
            rows.append({
                "case_id": case_id,
                "SeriesInstanceUID": str(row.SeriesInstanceUID),
                "sequence_path": str(output_path.resolve()),
                "token_count": token_count,
                "status": "cached",
            })
            continue
        try:
            series_path = _series_path(row, args.dicom_dir, manifest_dir)
            features, positions, metadata = model.encode_axial_sequence(
                series_path,
                max_tokens=args.max_tokens,
                inference_batch_size=args.inference_batch_size,
            )
            partial = output_path.with_suffix(".partial.npz")
            np.savez_compressed(
                partial,
                features=features.numpy().astype(storage_dtype),
                positions=positions.numpy().astype(np.float32),
                original_slices=np.asarray(metadata["original_slices"], dtype=np.int32),
            )
            partial.replace(output_path)
            rows.append({
                "case_id": case_id,
                "SeriesInstanceUID": str(row.SeriesInstanceUID),
                "sequence_path": str(output_path.resolve()),
                "token_count": int(features.shape[0]),
                "status": "built",
            })
        except Exception as exc:
            failures.append({
                "case_id": case_id,
                "SeriesInstanceUID": str(row.SeriesInstanceUID),
                "error": repr(exc),
            })
        print(
            f"[{position}/{len(selected)}] {case_id} valid={len(rows)} "
            f"failures={len(failures)}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("case_id").to_csv(
        args.output_dir / "manifest.csv", index=False
    )
    pd.DataFrame(
        failures, columns=["case_id", "SeriesInstanceUID", "error"]
    ).to_csv(args.output_dir / "failures.csv", index=False)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": "torchvision ResNet18 ImageNet1K V1, frozen",
        "context": f"ordered axial 2.5D windows {slice_offsets}",
        "slice_span": args.slice_span,
        "slice_offsets": slice_offsets,
        "sampling": "uniform inclusive endpoints",
        "max_tokens": args.max_tokens,
        "feature_dim": 512,
        "storage_dtype": args.storage_dtype,
        "outcome_independent": True,
        "series_manifest": str(args.series_manifest.resolve()),
        "n_valid": len(rows),
        "n_failures": len(failures),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
