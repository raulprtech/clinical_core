"""Build resumable frozen STU-Net-S masks and volumetric embeddings.

This technical-pilot runner is intentionally independent of survival splits.
It performs deterministic FP32-checkpoint inference with mixed-precision CUDA,
stores the 105-class probability accumulator on disk, and derives a 768D
embedding from the 256-channel encoder bottleneck:

    abdominal/kidney ROI pool || left-kidney pool || right-kidney pool

The three 256D vectors are concatenated and L2-normalized. Quantized variants
must use the same pooling implementation so their embedding drift is paired.
The runner also emits a predeclared 512D renal-moments candidate made from the
mean and standard deviation inside the same kidney bounding-box ROI. Both
representations are derived from one inference, so their comparison is paired.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import mmap
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pydicom
import psutil
import SimpleITK as sitk
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data/manifests/tcia_kirc/series_selected.csv"
DEFAULT_DICOM_ROOT = REPO_ROOT / "data/raw/tcia_kirc_dicom"
DEFAULT_MODEL_ROOT = REPO_ROOT / "data/models/stunet"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/embeddings/vision/stunet_fp32_pilot"

KIDNEY_LEFT_LABEL = 38
KIDNEY_RIGHT_LABEL = 39
EMBEDDING_DIM = 768
RENAL_MOMENTS_DIM = 512


def drop_memmap_page_cache(array: np.memmap) -> None:
    """Evict flushed pages so large probability files do not consume host RAM."""
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and hasattr(mapping, "madvise"):
        try:
            mapping.madvise(mmap.MADV_DONTNEED)
        except (OSError, ValueError):
            pass


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_spaced_rows(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Select deterministic cases across the observed slice-count range."""
    if limit <= 0 or len(frame) <= limit:
        return frame.copy()
    positions = np.linspace(0, len(frame) - 1, num=limit)
    indices = np.unique(np.rint(positions).astype(int))
    return frame.iloc[indices].copy()


def select_pilot_cases(
    manifest: Path,
    dicom_root: Path,
    limit: int,
    case_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    selected = pd.read_csv(manifest)
    required = {"case_id", "SeriesInstanceUID", "Modality"}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"Series manifest is missing columns: {sorted(missing)}")

    selected["case_id"] = selected["case_id"].astype(str).str.strip().str.upper()
    selected["Modality"] = selected["Modality"].astype(str).str.strip().str.upper()
    selected = selected[selected["Modality"] == "CT"].copy()
    if case_ids:
        requested = {str(case_id).strip().upper() for case_id in case_ids}
        selected = selected[selected["case_id"].isin(requested)].copy()

    count_column = "ImageCount_num" if "ImageCount_num" in selected else "ImageCount"
    selected[count_column] = pd.to_numeric(selected[count_column], errors="coerce")
    selected = selected.sort_values(
        [count_column, "case_id"], kind="stable", na_position="last"
    )
    complete = []
    for row in selected.itertuples(index=False):
        series_dir = dicom_root / row.case_id / str(row.SeriesInstanceUID)
        if (series_dir / ".complete.json").exists():
            complete.append(True)
        else:
            complete.append(False)
    selected = selected.loc[complete]
    return evenly_spaced_rows(selected, limit).reset_index(drop=True)


class PeakRSSMonitor:
    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakRSSMonitor":
        process = psutil.Process()

        def monitor() -> None:
            while not self._stop.wait(self.interval_seconds):
                try:
                    rss = process.memory_info().rss
                    for child in process.children(recursive=True):
                        rss += child.memory_info().rss
                    self.peak_bytes = max(self.peak_bytes, rss)
                except (psutil.Error, ProcessLookupError):
                    continue

        self.peak_bytes = process.memory_info().rss
        self._thread = threading.Thread(target=monitor, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


def dicom_geometry_metrics(files: Iterable[str]) -> dict[str, Any]:
    positions: list[float] = []
    for filename in files:
        dataset = pydicom.dcmread(
            filename,
            stop_before_pixels=True,
            specific_tags=["ImagePositionPatient", "ImageOrientationPatient"],
        )
        if not hasattr(dataset, "ImagePositionPatient") or not hasattr(
            dataset, "ImageOrientationPatient"
        ):
            continue
        orientation = np.asarray(dataset.ImageOrientationPatient, dtype=float)
        position = np.asarray(dataset.ImagePositionPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        positions.append(float(np.dot(position, normal)))
    if len(positions) < 3:
        return {
            "geometry_qc": "not_evaluable",
            "geometry_positions": len(positions),
        }
    ordered = np.sort(np.asarray(positions))
    unique = np.unique(np.round(ordered, decimals=5))
    duplicate_positions = len(ordered) - len(unique)
    gaps = np.diff(unique)
    median = float(np.median(gaps))
    minimum = float(gaps.min())
    maximum = float(gaps.max())
    gap_ratio = maximum / median if median > 0 else float("inf")
    minimum_ratio = minimum / median if median > 0 else 0.0
    uniform = duplicate_positions == 0 and gap_ratio <= 1.5 and minimum_ratio >= 0.5
    return {
        "geometry_qc": "pass" if uniform else "fail",
        "geometry_positions": len(positions),
        "duplicate_slice_positions": duplicate_positions,
        "slice_spacing_median_mm": median,
        "slice_spacing_min_mm": minimum,
        "slice_spacing_max_mm": maximum,
        "slice_gap_ratio": gap_ratio,
        "slice_min_gap_ratio": minimum_ratio,
    }


def convert_dicom_series(series_dir: Path, series_uid: str, output_path: Path) -> dict[str, Any]:
    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(series_dir)) or []
    if not ids:
        raise RuntimeError(f"No DICOM series found in {series_dir}")
    selected_uid = series_uid if series_uid in ids else ids[0]
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(series_dir), selected_uid)
    if not files:
        raise RuntimeError(f"No slices found for series {selected_uid}")
    geometry = dicom_geometry_metrics(files)
    if geometry["geometry_qc"] == "fail":
        raise RuntimeError(
            "DICOM geometry QC failed: "
            f"gap_ratio={geometry['slice_gap_ratio']:.3f}, "
            f"min_gap_ratio={geometry['slice_min_gap_ratio']:.3f}, "
            f"duplicates={geometry['duplicate_slice_positions']}"
        )
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    array = sitk.GetArrayViewFromImage(image)
    metrics = {
        "dicom_series_uid": selected_uid,
        "dicom_slices": len(files),
        "input_size_xyz": [int(value) for value in image.GetSize()],
        "input_spacing_xyz_mm": [float(value) for value in image.GetSpacing()],
        "hu_p005": float(np.percentile(array, 0.5)),
        "hu_p995": float(np.percentile(array, 99.5)),
        **geometry,
    }
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < np.iinfo(np.int16).min or maximum > np.iinfo(np.int16).max:
        raise RuntimeError(
            f"CT values [{minimum}, {maximum}] exceed lossless int16 storage"
        )
    image = sitk.Cast(image, sitk.sitkInt16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path), True)
    metrics["nifti_storage_dtype"] = "int16"
    return metrics


@dataclass
class PatchFeature:
    starts: tuple[int, int, int]
    values: np.ndarray


class STUNetRuntime:
    def __init__(
        self, model_root: Path, device: str, step_size: float, precision: str
    ):
        self.model_root = model_root.resolve()
        self.step_size = step_size
        self.precision = precision
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if self.device.type != "cuda":
            raise RuntimeError("The pilot runner currently requires CUDA inference")

        nnunet_root = self.model_root / "STU-Net/nnUNet-1.7.1"
        if str(nnunet_root) not in sys.path:
            sys.path.insert(0, str(nnunet_root))
        os.environ.setdefault("RESULTS_FOLDER", str(self.model_root))
        os.environ.setdefault("nnUNet_raw_data_base", str(self.model_root / "raw"))
        os.environ.setdefault("nnUNet_preprocessed", str(self.model_root / "preprocessed"))
        os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "data/cache/matplotlib"))

        from nnunet.training.model_restore import restore_model

        self.model_dir = (
            self.model_root
            / "nnUNet/3d_fullres/Task101_TotalSegmentator"
            / "STUNetTrainer_small__nnUNetPlansv2.1/fold_0"
        )
        self.checkpoint_path = self.model_dir / "small_ep4k.model"
        metadata_path = self.model_dir / "small_ep4k.model.pkl"
        if not self.checkpoint_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("STU-Net-S checkpoint or metadata is missing")

        self.trainer = restore_model(str(metadata_path), fp16=True)
        self.trainer.initialize_network()
        # load_checkpoint_ram otherwise calls initialize(False), which also
        # constructs an optimizer and runs torchinfo on a 128^3 tensor.
        self.trainer.was_initialized = True
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        self.trainer.load_checkpoint_ram(checkpoint, train=False)
        self.network = self.trainer.network
        self.network.do_ds = False
        self.network.eval()
        self.network.to(self.device)
        self.patch_size = np.asarray(self.trainer.patch_size, dtype=int)
        self.num_classes = int(self.trainer.num_classes)
        if self.num_classes != 105:
            raise ValueError(f"Expected 105 classes, found {self.num_classes}")

    def preprocess(self, nifti_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        data, _, properties = self.trainer.preprocess_patient([str(nifti_path)])
        return data.astype(np.float32, copy=False), properties

    def _stream_probabilities(
        self,
        data: np.ndarray,
        work_dir: Path,
        logit_sketch_path: Path | None = None,
    ) -> tuple[np.ndarray, list[PatchFeature], dict[str, Any]]:
        from batchgenerators.augmentations.utils import pad_nd_image

        padded, slicer = pad_nd_image(
            data, self.patch_size, "constant", {"constant_values": 0}, True, None
        )
        spatial_shape = tuple(int(value) for value in padded.shape[1:])
        steps = self.network._compute_steps_for_sliding_window(
            self.patch_size, spatial_shape, self.step_size
        )
        starts = [
            (int(x), int(y), int(z))
            for x in steps[0]
            for y in steps[1]
            for z in steps[2]
        ]

        gaussian = self.network._get_gaussian(self.patch_size).astype(np.float16)
        nonzero = gaussian[gaussian != 0]
        gaussian[gaussian == 0] = nonzero.min()
        gaussian_gpu = torch.from_numpy(gaussian).to(self.device)

        work_dir.mkdir(parents=True, exist_ok=True)
        probabilities_path = work_dir / "probabilities.f16.memmap"
        counts_path = work_dir / "counts.f16.memmap"
        probabilities = np.memmap(
            probabilities_path,
            mode="w+",
            dtype=np.float16,
            shape=(self.num_classes, *spatial_shape),
        )
        counts = np.memmap(
            counts_path, mode="w+", dtype=np.float16, shape=spatial_shape
        )
        # Fresh w+ memmaps are backed by newly truncated, zero-filled files.
        # Touching every page here can fill the Linux dirty-page cache with
        # several GiB before inference starts and trigger the OOM killer.

        captured: list[np.ndarray] = []

        def bottleneck_hook(_module: object, _inputs: object, output: torch.Tensor) -> None:
            captured.append(output.detach().to("cpu", dtype=torch.float32).numpy()[0])

        hook = self.network.conv_blocks_context[-1].register_forward_hook(bottleneck_hook)
        patch_features: list[PatchFeature] = []
        logit_sketches: list[np.ndarray] = []
        torch.cuda.reset_peak_memory_stats(self.device)
        inference_start = time.perf_counter()
        try:
            for position, (x, y, z) in enumerate(starts, 1):
                patch_np = padded[
                    None,
                    :,
                    x : x + self.patch_size[0],
                    y : y + self.patch_size[1],
                    z : z + self.patch_size[2],
                ]
                patch = torch.from_numpy(patch_np).to(self.device, non_blocking=True)
                captured.clear()
                autocast_context = (
                    torch.amp.autocast("cuda", dtype=torch.float16)
                    if self.precision == "amp"
                    else nullcontext()
                )
                with torch.inference_mode(), autocast_context:
                    logits = self.network(patch)
                    if logit_sketch_path is not None:
                        logit_sketches.append(
                            logits[0, :, ::16, ::16, ::16]
                            .detach()
                            .to("cpu", dtype=torch.float16)
                            .numpy()
                        )
                    softmax = torch.softmax(logits, dim=1)
                    weighted = softmax * gaussian_gpu[None, None]
                weighted_np = weighted[0].to("cpu", dtype=torch.float16).numpy()
                view = np.s_[
                    :,
                    x : x + self.patch_size[0],
                    y : y + self.patch_size[1],
                    z : z + self.patch_size[2],
                ]
                probabilities[view] += weighted_np
                counts[
                    x : x + self.patch_size[0],
                    y : y + self.patch_size[1],
                    z : z + self.patch_size[2],
                ] += gaussian
                if len(captured) != 1:
                    raise RuntimeError(
                        f"Expected one bottleneck activation, captured {len(captured)}"
                    )
                patch_features.append(
                    PatchFeature((x, y, z), np.array(captured[0], copy=True))
                )
                del patch, logits, softmax, weighted, weighted_np
                probabilities.flush()
                counts.flush()
                drop_memmap_page_cache(probabilities)
                drop_memmap_page_cache(counts)
        finally:
            hook.remove()

        probabilities.flush()
        counts.flush()
        drop_memmap_page_cache(probabilities)
        drop_memmap_page_cache(counts)
        if logit_sketch_path is not None:
            logit_sketch_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                logit_sketch_path,
                logits=np.stack(logit_sketches),
                sample_stride=np.asarray([16, 16, 16], dtype=np.int16),
                tile_starts=np.asarray(starts, dtype=np.int32),
            )
        segmentation = np.zeros(spatial_shape, dtype=np.uint8)
        chunk_size = max(1, min(4, spatial_shape[0]))
        for lower in range(0, spatial_shape[0], chunk_size):
            upper = min(spatial_shape[0], lower + chunk_size)
            denominator = np.asarray(counts[lower:upper], dtype=np.float32)
            denominator = np.maximum(denominator, np.finfo(np.float16).tiny)
            chunk = np.asarray(probabilities[:, lower:upper], dtype=np.float32)
            chunk /= denominator[None]
            segmentation[lower:upper] = np.argmax(chunk, axis=0).astype(np.uint8)
            del chunk, denominator

        del probabilities, counts
        probabilities_path.unlink(missing_ok=True)
        counts_path.unlink(missing_ok=True)
        unpadded = segmentation[tuple(slicer[1:])]
        metrics = {
            "n_tiles": len(starts),
            "preprocessed_shape_cxyz": [int(value) for value in data.shape],
            "padded_shape_cxyz": [int(value) for value in padded.shape],
            "patch_size_xyz": [int(value) for value in self.patch_size],
            "step_size": float(self.step_size),
            "inference_seconds": time.perf_counter() - inference_start,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
            "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(self.device) / 2**30,
            "probability_memmap_gib": (
                self.num_classes * math.prod(spatial_shape) * np.dtype(np.float16).itemsize
            )
            / 2**30,
        }
        return unpadded, patch_features, metrics

    @staticmethod
    def _kidney_roi(segmentation: np.ndarray, margin_voxels: int) -> tuple[slice, slice, slice]:
        kidney = (segmentation == KIDNEY_LEFT_LABEL) | (
            segmentation == KIDNEY_RIGHT_LABEL
        )
        coordinates = np.argwhere(kidney)
        if coordinates.size == 0:
            return tuple(slice(0, size) for size in segmentation.shape)  # type: ignore[return-value]
        lower = np.maximum(coordinates.min(axis=0) - margin_voxels, 0)
        upper = np.minimum(coordinates.max(axis=0) + 1 + margin_voxels, segmentation.shape)
        return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))  # type: ignore[return-value]

    def _pool_embedding_variants(
        self,
        segmentation: np.ndarray,
        patch_features: list[PatchFeature],
        margin_voxels: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        roi = self._kidney_roi(segmentation, margin_voxels)
        roi_lower = np.asarray([part.start for part in roi], dtype=int)
        roi_upper = np.asarray([part.stop for part in roi], dtype=int)
        numerators = np.zeros((3, 256), dtype=np.float64)
        squared_numerators = np.zeros((3, 256), dtype=np.float64)
        denominators = np.zeros(3, dtype=np.float64)
        gaussian = self.network._get_gaussian(self.patch_size)

        for patch_feature in patch_features:
            starts = np.asarray(patch_feature.starts, dtype=int)
            stops = starts + self.patch_size
            feature = patch_feature.values.astype(np.float32, copy=False)
            tile_seg = segmentation[
                starts[0] : stops[0], starts[1] : stops[1], starts[2] : stops[2]
            ]
            tile_roi = np.zeros(tile_seg.shape, dtype=np.float32)
            overlap_lower = np.maximum(starts, roi_lower)
            overlap_upper = np.minimum(stops, roi_upper)
            if np.all(overlap_upper > overlap_lower):
                local_lower = overlap_lower - starts
                local_upper = overlap_upper - starts
                tile_roi[
                    local_lower[0] : local_upper[0],
                    local_lower[1] : local_upper[1],
                    local_lower[2] : local_upper[2],
                ] = 1
            masks = np.stack(
                [
                    tile_roi,
                    (tile_seg == KIDNEY_LEFT_LABEL).astype(np.float32),
                    (tile_seg == KIDNEY_RIGHT_LABEL).astype(np.float32),
                ]
            )
            masks_small = F.interpolate(
                torch.from_numpy(masks[:, None]),
                size=feature.shape[1:],
                mode="area",
            )[:, 0].numpy()
            feature_gaussian = F.interpolate(
                torch.from_numpy(gaussian)[None, None],
                size=feature.shape[1:],
                mode="trilinear",
                align_corners=False,
            )[0, 0].numpy()
            for branch in range(3):
                weight = masks_small[branch] * feature_gaussian
                denominator = float(weight.sum())
                if denominator > 0:
                    numerators[branch] += (feature * weight[None]).sum(axis=(1, 2, 3))
                    squared_numerators[branch] += (
                        feature.astype(np.float64) ** 2 * weight[None]
                    ).sum(axis=(1, 2, 3))
                    denominators[branch] += denominator

        pooled = np.zeros((3, 256), dtype=np.float32)
        pooled_std = np.zeros((3, 256), dtype=np.float32)
        valid = denominators > 0
        pooled[valid] = (numerators[valid] / denominators[valid, None]).astype(np.float32)
        second_moments = np.zeros((3, 256), dtype=np.float64)
        second_moments[valid] = squared_numerators[valid] / denominators[valid, None]
        variances = np.maximum(second_moments - pooled.astype(np.float64) ** 2, 0.0)
        pooled_std[valid] = np.sqrt(variances[valid]).astype(np.float32)

        mean_768 = pooled.reshape(-1)
        renal_moments_512 = np.concatenate([pooled[0], pooled_std[0]])
        variants = {
            "mean_768": mean_768,
            "renal_moments_512": renal_moments_512,
        }
        for embedding in variants.values():
            norm = float(np.linalg.norm(embedding))
            if norm > 0:
                embedding /= norm
        metrics = {
            "embedding_dims": {
                name: int(embedding.size) for name, embedding in variants.items()
            },
            "embedding_l2_norms": {
                name: float(np.linalg.norm(embedding))
                for name, embedding in variants.items()
            },
            "pool_denominators": denominators.tolist(),
            "kidney_roi_preprocessed": [
                [int(part.start), int(part.stop)] for part in roi
            ],
        }
        if mean_768.size != EMBEDDING_DIM:
            raise ValueError(f"Expected {EMBEDDING_DIM}D embedding, got {mean_768.size}")
        if renal_moments_512.size != RENAL_MOMENTS_DIM:
            raise ValueError(
                f"Expected {RENAL_MOMENTS_DIM}D renal moments, "
                f"got {renal_moments_512.size}"
            )
        return variants, metrics

    def _pool_embedding(
        self,
        segmentation: np.ndarray,
        patch_features: list[PatchFeature],
        margin_voxels: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Backward-compatible historical 768D pooling API."""
        variants, metrics = self._pool_embedding_variants(
            segmentation, patch_features, margin_voxels
        )
        metrics = {
            **metrics,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_l2_norm": metrics["embedding_l2_norms"]["mean_768"],
        }
        return variants["mean_768"], metrics

    def run_case_variants(
        self,
        nifti_path: Path,
        segmentation_path: Path,
        work_dir: Path,
        roi_margin_mm: float,
        logit_sketch_path: Path | None = None,
        export_order: int = 1,
        preprocessed: tuple[np.ndarray, dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        from nnunet.inference.segmentation_export import save_segmentation_nifti

        started = time.perf_counter()
        preprocess_started = time.perf_counter()
        if preprocessed is None:
            data, properties = self.preprocess(nifti_path)
        else:
            data, properties = preprocessed
        preprocess_seconds = time.perf_counter() - preprocess_started
        target_spacing = np.asarray(properties["spacing_after_resampling"], dtype=float)
        margin_voxels = int(math.ceil(roi_margin_mm / float(target_spacing.min())))
        segmentation, patch_features, inference_metrics = self._stream_probabilities(
            data, work_dir, logit_sketch_path=logit_sketch_path
        )
        embeddings, embedding_metrics = self._pool_embedding_variants(
            segmentation, patch_features, margin_voxels
        )
        export_started = time.perf_counter()
        segmentation_path.parent.mkdir(parents=True, exist_ok=True)
        save_segmentation_nifti(
            segmentation,
            str(segmentation_path),
            properties,
            order=export_order,
            force_separate_z=None,
            order_z=0,
            verbose=True,
        )
        export_seconds = time.perf_counter() - export_started
        original = sitk.ReadImage(str(segmentation_path))
        original_array = sitk.GetArrayViewFromImage(original)
        voxel_ml = float(np.prod(original.GetSpacing()) / 1000.0)
        left_voxels = int(np.count_nonzero(original_array == KIDNEY_LEFT_LABEL))
        right_voxels = int(np.count_nonzero(original_array == KIDNEY_RIGHT_LABEL))
        confidence = (float(left_voxels > 0) + float(right_voxels > 0)) / 2.0
        metrics = {
            **inference_metrics,
            **embedding_metrics,
            "preprocess_seconds": preprocess_seconds,
            "export_seconds": export_seconds,
            "segmentation_export_order": int(export_order),
            "total_seconds": time.perf_counter() - started,
            "target_spacing_xyz_mm": target_spacing.tolist(),
            "kidney_left_voxels": left_voxels,
            "kidney_right_voxels": right_voxels,
            "kidney_left_volume_ml": left_voxels * voxel_ml,
            "kidney_right_volume_ml": right_voxels * voxel_ml,
            "vision_confidence": confidence,
        }
        del data, segmentation, patch_features, original_array
        gc.collect()
        torch.cuda.empty_cache()
        return embeddings, metrics

    def run_case(
        self,
        nifti_path: Path,
        segmentation_path: Path,
        work_dir: Path,
        roi_margin_mm: float,
        logit_sketch_path: Path | None = None,
        export_order: int = 1,
        preprocessed: tuple[np.ndarray, dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run one case and preserve the historical 768D return contract."""
        embeddings, metrics = self.run_case_variants(
            nifti_path,
            segmentation_path,
            work_dir,
            roi_margin_mm,
            logit_sketch_path=logit_sketch_path,
            export_order=export_order,
            preprocessed=preprocessed,
        )
        return embeddings["mean_768"], {
            **metrics,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_l2_norm": metrics["embedding_l2_norms"]["mean_768"],
        }


def marker_has_variants(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return set(payload.get("embedding_variants", {})) == {
        "mean_768",
        "renal_moments_512",
    }


def rebuild_outputs(output_root: Path, selected: pd.DataFrame) -> tuple[int, int]:
    rows_by_variant: dict[str, list[dict[str, Any]]] = {
        "mean_768": [],
        "renal_moments_512": [],
    }
    metrics_rows: list[dict[str, Any]] = []
    selected_ids = set(selected["case_id"].astype(str))
    for marker in sorted((output_root / "cases").glob("*/complete.json")):
        payload = json.loads(marker.read_text())
        if payload.get("case_id") not in selected_ids:
            continue
        embeddings = payload.get("embedding_variants")
        if embeddings is None and "embedding" in payload:
            embeddings = {"mean_768": payload["embedding"]}
        for variant, embedding in embeddings.items():
            if variant not in rows_by_variant:
                continue
            row = {
                "case_id": payload["case_id"],
                "vision_available": 1,
                "vision_confidence": payload["metrics"]["vision_confidence"],
                "embedding_source": (
                    f"{payload['variant']}_bottleneck_{variant}"
                ),
                "SeriesInstanceUID": payload["SeriesInstanceUID"],
            }
            row.update(
                {f"z{idx:03d}": float(value) for idx, value in enumerate(embedding)}
            )
            rows_by_variant[variant].append(row)
        metrics_rows.append(
            {
                "case_id": payload["case_id"],
                "SeriesInstanceUID": payload["SeriesInstanceUID"],
                **payload["input"],
                **payload["metrics"],
            }
        )

    filenames = {
        "mean_768": "stunet_s_fp32_embeddings_768.csv",
        "renal_moments_512": "stunet_s_fp32_renal_moments_512.csv",
    }
    for variant, rows in rows_by_variant.items():
        frame = pd.DataFrame(rows)
        if rows:
            frame = frame.sort_values("case_id")
        frame.to_csv(output_root / filenames[variant], index=False)
    metrics_frame = pd.DataFrame(metrics_rows)
    if metrics_rows:
        metrics_frame = metrics_frame.sort_values("case_id")
    metrics_frame.to_csv(output_root / "metrics.csv", index=False)

    failures = list((output_root / "cases").glob("*/failure.json"))
    failure_rows = [
        payload
        for path in failures
        if (payload := json.loads(path.read_text())).get("case_id") in selected_ids
    ]
    pd.DataFrame(failure_rows).to_csv(output_root / "failures.csv", index=False)
    return len(rows_by_variant["mean_768"]), len(failure_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dicom-root", type=Path, default=DEFAULT_DICOM_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument(
        "--case-id-file",
        type=Path,
        help="CSV containing a case_id column; combined with --case-ids if provided",
    )
    parser.add_argument("--device", choices=["auto", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["fp32", "amp"], default="amp")
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--roi-margin-mm", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.step_size <= 1:
        raise ValueError("--step-size must be in (0, 1]")
    args.output_root.mkdir(parents=True, exist_ok=True)
    requested_case_ids = list(args.case_ids or [])
    if args.case_id_file is not None:
        case_id_frame = pd.read_csv(args.case_id_file)
        if "case_id" not in case_id_frame:
            raise ValueError("--case-id-file must contain a case_id column")
        requested_case_ids.extend(case_id_frame["case_id"].astype(str).tolist())
    selected = select_pilot_cases(
        args.series_manifest,
        args.dicom_root,
        args.limit,
        requested_case_ids or None,
    )
    if selected.empty:
        raise RuntimeError("No complete CT cases matched the pilot selection")
    selected.to_csv(args.output_root / "pilot_cohort.csv", index=False)
    print(
        "Selected pilot: "
        + ", ".join(
            f"{row.case_id}({int(row.ImageCount_num)} slices)"
            for row in selected.itertuples(index=False)
        ),
        flush=True,
    )

    cached = selected.apply(
        lambda row: (
            args.output_root / "cases" / row["case_id"] / "complete.json"
        ).exists()
        and marker_has_variants(
            args.output_root / "cases" / row["case_id"] / "complete.json"
        )
        and (
            args.output_root
            / "cases"
            / row["case_id"]
            / f"{row['case_id']}_stunet_seg.nii.gz"
        ).exists(),
        axis=1,
    )
    if bool(cached.all()) and not args.force:
        valid, failed = rebuild_outputs(args.output_root, selected)
        print(f"Pilot already complete: valid={valid}, failed={failed}", flush=True)
        return 0 if valid == len(selected) and failed == 0 else 1

    runtime = STUNetRuntime(
        args.model_root, args.device, args.step_size, args.precision
    )
    checkpoint_sha256 = sha256_file(runtime.checkpoint_path)
    started_at = time.time()
    for position, row in enumerate(selected.itertuples(index=False), 1):
        case_id = row.case_id
        case_root = args.output_root / "cases" / case_id
        marker = case_root / "complete.json"
        segmentation_path = case_root / f"{case_id}_stunet_seg.nii.gz"
        if (
            marker_has_variants(marker)
            and segmentation_path.exists()
            and not args.force
        ):
            print(f"[{position}/{len(selected)}] {case_id}: cached", flush=True)
            continue
        case_root.mkdir(parents=True, exist_ok=True)
        failure_path = case_root / "failure.json"
        failure_path.unlink(missing_ok=True)
        series_dir = args.dicom_root / case_id / str(row.SeriesInstanceUID)
        nifti_path = case_root / "input" / f"{case_id}_0000.nii.gz"
        work_dir = case_root / "work"
        print(f"[{position}/{len(selected)}] {case_id}: converting DICOM", flush=True)
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
            input_metrics = convert_dicom_series(
                series_dir, str(row.SeriesInstanceUID), nifti_path
            )
            with PeakRSSMonitor() as memory:
                embeddings, metrics = runtime.run_case_variants(
                    nifti_path,
                    segmentation_path,
                    work_dir,
                    args.roi_margin_mm,
                )
            metrics["peak_rss_gib"] = memory.peak_bytes / 2**30
            payload = {
                "schema_version": 2,
                "case_id": case_id,
                "SeriesInstanceUID": str(row.SeriesInstanceUID),
                "variant": f"stunet_s_weights_fp32_{args.precision}",
                "checkpoint": str(runtime.checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "input": input_metrics,
                "metrics": metrics,
                "embedding_variants": {
                    name: embedding.astype(float).tolist()
                    for name, embedding in embeddings.items()
                },
                "segmentation_path": str(segmentation_path.resolve()),
            }
            atomic_json_dump(payload, marker)
            shutil.rmtree(work_dir, ignore_errors=True)
            print(
                f"[{position}/{len(selected)}] {case_id}: done "
                f"{metrics['total_seconds']:.1f}s, tiles={metrics['n_tiles']}, "
                f"RSS={metrics['peak_rss_gib']:.2f}GiB, "
                f"GPU={metrics['gpu_peak_allocated_gib']:.2f}GiB",
                flush=True,
            )
        except Exception as exc:
            shutil.rmtree(work_dir, ignore_errors=True)
            payload = {
                "case_id": case_id,
                "SeriesInstanceUID": str(row.SeriesInstanceUID),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            atomic_json_dump(payload, failure_path)
            print(f"[{position}/{len(selected)}] {case_id}: FAILED {exc!r}", flush=True)
        rebuild_outputs(args.output_root, selected)

    valid, failed = rebuild_outputs(args.output_root, selected)
    provenance = {
        "schema_version": 2,
        "built_at_unix": time.time(),
        "elapsed_seconds": time.time() - started_at,
        "series_manifest": str(args.series_manifest.resolve()),
        "dicom_root": str(args.dicom_root.resolve()),
        "model_root": str(args.model_root.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "selected_cases": selected["case_id"].tolist(),
        "n_selected": len(selected),
        "n_valid": valid,
        "n_failed": failed,
        "embedding_definitions": {
            "mean_768": (
                "GAP256(kidney_bbox+margin)||GAP256(left)||GAP256(right), L2"
            ),
            "renal_moments_512": (
                "mean256(kidney_bbox+margin)||std256(kidney_bbox+margin), L2"
            ),
        },
        "embedding_dims": {
            "mean_768": EMBEDDING_DIM,
            "renal_moments_512": RENAL_MOMENTS_DIM,
        },
        "step_size": args.step_size,
        "tta": False,
        "precision": args.precision,
        "mixed_precision": args.precision == "amp",
        "probability_accumulation": "disk_memmap_float16",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(runtime.device),
    }
    atomic_json_dump(provenance, args.output_root / "provenance.json")
    print(f"Pilot complete: valid={valid}, failed={failed}", flush=True)
    return 0 if valid == len(selected) and failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
