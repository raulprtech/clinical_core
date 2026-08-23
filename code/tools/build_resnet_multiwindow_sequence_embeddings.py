"""Build frozen ResNet18 2.5D axial sequences with fixed CT multi-window fusion.

The baseline extractor remains untouched. CT tokens fuse three predeclared HU
windows in feature space; MR uses the legacy single percentile-normalized pass.
Patient-level files are resumable and contain no outcomes.
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
TOOLS_ROOT = Path(__file__).resolve().parent
for candidate in (CODE_ROOT, TOOLS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from components.adapters.ingestion.vision.models.resnet_multiwindow import (  # noqa: E402
    DEFAULT_CT_WINDOWS,
    VisionResNet18_2p5DMultiWindow,
)
from build_resnet_sequence_embeddings import _series_path  # noqa: E402


PRESET_NAME = "renal_multi3_v1"


def valid_case_cache(path: Path, max_tokens: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            features = payload["features"]
            positions = payload["positions"]
            preset = str(payload["ct_window_preset"].item())
        return (
            features.ndim == 2
            and features.shape[1] == 512
            and 1 <= features.shape[0] <= max_tokens
            and positions.shape == (features.shape[0],)
            and np.isfinite(features).all()
            and np.isfinite(positions).all()
            and preset == PRESET_NAME
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
        default=Path(
            "data/embeddings/vision/resnet18_2p5d_sequences_multiwindow3"
        ),
    )
    parser.add_argument("--weights-dir", type=Path, default=Path("data/models/torch"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument(
        "--storage-dtype", choices=["float16", "float32"], default="float16"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.max_tokens < 2:
        raise ValueError("--max-tokens must be at least 2")
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
    model = VisionResNet18_2p5DMultiWindow(
        output_dim=768,
        use_imagenet_weights=True,
        image_size=224,
        window_low=-150,
        window_high=250,
        ct_windows=DEFAULT_CT_WINDOWS,
        min_slices=16,
        slice_offsets=[-1, 0, 1],
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
        if not args.force and valid_case_cache(output_path, args.max_tokens):
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
                ct_window_preset=np.asarray(PRESET_NAME),
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
        "context": "ordered axial 2.5D neighbors [-1, 0, 1]",
        "ct_window_preset": PRESET_NAME,
        "ct_windows": [list(bounds) for bounds in DEFAULT_CT_WINDOWS],
        "ct_window_feature_fusion": "equal_mean_then_l2",
        "mr_processing_changed": False,
        "sampling": "uniform inclusive endpoints",
        "max_tokens": args.max_tokens,
        "feature_dim": 512,
        "storage_dtype": args.storage_dtype,
        "outcome_independent": True,
        "series_manifest": str(args.series_manifest.resolve()),
        "n_valid": len(rows),
        "n_failures": len(failures),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
