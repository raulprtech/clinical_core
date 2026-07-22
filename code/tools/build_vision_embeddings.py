"""Build resumable 768D vision caches for the three VISION-L0 variants."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.model_utils import verify_ingestion_contract  # noqa: E402
from core.registry import get_vision_conn  # noqa: E402


VARIANTS = (
    "vision_resnet18_2d",
    "vision_resnet50_2d",
    "vision_resnet18_2p5d",
)


def write_cache(rows: list[dict], path: Path) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("case_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-manifest", required=True, type=Path)
    parser.add_argument("--dicom-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/embeddings/vision"))
    parser.add_argument("--weights-dir", type=Path, default=Path("data/models/torch"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = pd.read_csv(args.series_manifest)
    required = {"case_id", "SeriesInstanceUID"}
    if not required.issubset(selected.columns):
        raise ValueError(f"Series manifest is missing {sorted(required - set(selected.columns))}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for variant in args.variants:
        output_path = args.output_dir / f"{variant}_embeddings_768.csv"
        existing_rows = []
        if output_path.exists() and not args.force:
            existing_rows = pd.read_csv(output_path).to_dict("records")
        done = {str(row["case_id"]).strip().upper() for row in existing_rows}
        rows = existing_rows
        model = get_vision_conn(
            variant,
            output_dim=768,
            use_imagenet_weights=True,
            aggregation="mean3",
            image_size=224,
            window_low=-150,
            window_high=250,
            min_slices=16,
            slice_offsets=[-1, 0, 1],
            projection_seed=2026,
            device=args.device,
            weights_dir=args.weights_dir.resolve(),
        )
        pending = selected[
            ~selected["case_id"].astype(str).str.upper().isin(done)
        ]
        print(
            f"[{variant}] total={len(selected)} cached={len(done)} pending={len(pending)}",
            flush=True,
        )
        for position, row in enumerate(pending.itertuples(index=False), 1):
            case_id = str(row.case_id).strip().upper()
            series_dir = args.dicom_dir / case_id / str(row.SeriesInstanceUID)
            if not (series_dir / ".complete.json").exists():
                failures.append({
                    "variant": variant, "case_id": case_id,
                    "series_dir": str(series_dir), "error": "download_not_complete",
                })
                continue
            try:
                embedding, confidence = model.encode(series_dir)
                check = verify_ingestion_contract(
                    embedding.unsqueeze(0),
                    torch.tensor([[confidence]], dtype=torch.float32),
                    verbose=False,
                )
                if not check["contract_satisfied"]:
                    raise ValueError(f"VISION-IN contract failed: {check}")
                record = {
                    "case_id": case_id,
                    "vision_available": 1,
                    "vision_confidence": float(confidence),
                    "embedding_source": variant,
                    "SeriesInstanceUID": str(row.SeriesInstanceUID),
                }
                record.update({
                    f"z{index:03d}": float(value)
                    for index, value in enumerate(embedding.numpy())
                })
                rows.append(record)
            except Exception as exc:
                failures.append({
                    "variant": variant, "case_id": case_id,
                    "series_dir": str(series_dir), "error": repr(exc),
                })
            if position % args.checkpoint_every == 0:
                write_cache(rows, output_path)
                print(
                    f"[{variant}] {position}/{len(pending)} processed; valid={len(rows)}",
                    flush=True,
                )
        write_cache(rows, output_path)
        print(f"[{variant}] saved {len(rows)} embeddings to {output_path}", flush=True)

    failures_path = args.output_dir / "failures.csv"
    pd.DataFrame(
        failures,
        columns=["variant", "case_id", "series_dir", "error"],
    ).to_csv(failures_path, index=False)
    sidecar = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "series_manifest": str(args.series_manifest.resolve()),
        "dicom_dir": str(args.dicom_dir.resolve()),
        "variants": args.variants,
        "embedding_dim": 768,
        "aggregation": "mean3",
        "imagenet_weights": True,
        "projection_seed": 2026,
        "n_failures": len(failures),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(sidecar, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
