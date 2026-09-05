"""Build matched frozen ResNet50 2.5D caches for ImageNet/RadImageNet.

Both variants use the same selected series, 64 axial tokens, renal CT window,
2.5D neighbours, and fixed 2048 -> 512 Gaussian projection. The only encoder
contract difference is the official pretraining weights and their prescribed
input normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CODE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for candidate in (CODE_ROOT, TOOLS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from build_resnet_sequence_embeddings import _series_path  # noqa: E402
from components.adapters.ingestion.vision.models.resnet_multiview import (  # noqa: E402
    VisionResNet50_2p5D,
)


RADIMAGENET_SOURCE = (
    "https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/"
    "view?usp=sharing"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_radimagenet_backbone(checkpoint: Path) -> nn.Module:
    """Strictly load the authors' PyTorch ResNet50 backbone checkpoint."""
    from torchvision import models

    if not checkpoint.is_file():
        raise FileNotFoundError(f"RadImageNet checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("RadImageNet checkpoint must be a non-empty state dict")
    prefix = "backbone."
    if not all(isinstance(key, str) and key.startswith(prefix) for key in payload):
        raise ValueError("Unexpected RadImageNet state-dict namespace")
    state = {key[len(prefix):]: value for key, value in payload.items()}
    base = models.resnet50(weights=None)
    backbone = nn.Sequential(*list(base.children())[:9])
    backbone.load_state_dict(state, strict=True)
    return backbone.eval()


def valid_case_cache(
    path: Path, max_tokens: int, feature_dim: int, encoder_id: str
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            features = payload["features"]
            positions = payload["positions"]
            cached_encoder = str(payload["encoder_id"].item())
        return (
            features.ndim == 2
            and features.shape[1] == feature_dim
            and 1 <= features.shape[0] <= max_tokens
            and positions.shape == (features.shape[0],)
            and np.isfinite(features).all()
            and np.isfinite(positions).all()
            and cached_encoder == encoder_id
        )
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-manifest", required=True, type=Path)
    parser.add_argument("--dicom-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--pretraining", required=True, choices=("imagenet", "radimagenet")
    )
    parser.add_argument(
        "--radimagenet-checkpoint",
        type=Path,
        default=Path("data/models/RadImageNet_pytorch/ResNet50.pt"),
    )
    parser.add_argument("--weights-dir", type=Path, default=Path("data/models/torch"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--projection-seed", type=int, default=2026)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument(
        "--storage-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.max_tokens < 2 or args.projection_dim < 1:
        raise ValueError("max tokens must be >=2 and projection dim must be positive")
    selected = pd.read_csv(args.series_manifest)
    required = {"case_id", "SeriesInstanceUID"}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"Series manifest is missing {sorted(missing)}")
    selected["case_id"] = selected["case_id"].astype(str).str.strip().str.upper()
    if selected["case_id"].duplicated().any():
        raise ValueError("Series manifest contains duplicate case_id values")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected.iloc[: args.limit].copy()

    checkpoint_sha256 = None
    if args.pretraining == "radimagenet":
        backbone = load_radimagenet_backbone(args.radimagenet_checkpoint)
        checkpoint_sha256 = file_sha256(args.radimagenet_checkpoint)
        normalization = {"mean": [0.5] * 3, "std": [0.5] * 3}
        encoder_id = f"resnet50_radimagenet_{checkpoint_sha256[:12]}"
    else:
        backbone = None
        normalization = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        encoder_id = "resnet50_imagenet1k_v2_torchvision"

    model = VisionResNet50_2p5D(
        output_dim=args.projection_dim,
        use_imagenet_weights=args.pretraining == "imagenet",
        image_size=224,
        window_low=-150,
        window_high=250,
        min_slices=16,
        slice_offsets=[-1, 0, 1],
        projection_seed=args.projection_seed,
        input_mean=normalization["mean"],
        input_std=normalization["std"],
        device=args.device,
        weights_dir=args.weights_dir.resolve(),
        backbone=backbone,
        feature_dim=2048,
    )
    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.series_manifest.resolve().parent
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    storage_dtype = np.float16 if args.storage_dtype == "float16" else np.float32
    for position, row in enumerate(selected.itertuples(index=False), 1):
        case_id = str(row.case_id)
        output_path = cases_dir / f"{case_id}.npz"
        if not args.force and valid_case_cache(
            output_path, args.max_tokens, args.projection_dim, encoder_id
        ):
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
                encoder_id=np.asarray(encoder_id),
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
        "encoder": f"torchvision ResNet50 {args.pretraining}, frozen",
        "encoder_id": encoder_id,
        "pretraining": args.pretraining,
        "preprocessing": {
            "ct_window": [-150, 250],
            "mr_window": "per-slice percentiles [1, 99]",
            "normalization": normalization,
            "normalization_source": (
                "official RadImageNet PyTorch example: pixels scaled to [-1, 1]"
                if args.pretraining == "radimagenet"
                else "torchvision ImageNet weights contract"
            ),
        },
        "context": "ordered axial 2.5D neighbours [-1, 0, 1]",
        "sampling": "uniform inclusive endpoints",
        "max_tokens": args.max_tokens,
        "raw_feature_dim": 2048,
        "feature_dim": args.projection_dim,
        "projection": "NumPy PCG64 fixed Gaussian, scale=1/sqrt(output_dim)",
        "projection_seed": args.projection_seed,
        "storage_dtype": args.storage_dtype,
        "outcome_independent": True,
        "series_manifest": str(args.series_manifest.resolve()),
        "radimagenet_checkpoint": (
            str(args.radimagenet_checkpoint.resolve())
            if args.pretraining == "radimagenet"
            else None
        ),
        "radimagenet_checkpoint_sha256": checkpoint_sha256,
        "radimagenet_official_source": (
            RADIMAGENET_SOURCE if args.pretraining == "radimagenet" else None
        ),
        "n_valid": len(rows),
        "n_failures": len(failures),
        "limit": args.limit,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
