"""Preliminary paired FP32/PTQ/TurboConv evaluation for frozen STU-Net-S.

Protocol:
  * 25 geometry-QC-pass CT cases calibrate activation ranges.
  * 50 disjoint geometry-QC-pass CT cases evaluate FP32, W4A8 PTQ and W4A8
    TurboConv.
  * paired drift is measured for sampled logits, 768D embeddings and masks.

The quantized variants use fake quantization to isolate numerical accuracy.
Reported model-size compression is theoretical; latency includes Python/PyTorch
fake-quant and Hadamard overhead and is not an optimized integer-kernel result.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from components.adapters.ingestion.vision.models.turboconv import (  # noqa: E402
    FakeQuantConv3d,
    QuantizationSpec,
    deterministic_signs,
    is_power_of_two,
    replace_conv3d_modules,
    rotate_channels,
    weight_error_summary,
)
from tools.build_stunet_embeddings import (  # noqa: E402
    DEFAULT_DICOM_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_ROOT,
    PeakRSSMonitor,
    STUNetRuntime,
    atomic_json_dump,
    convert_dicom_series,
    sha256_file,
)


DEFAULT_GEOMETRY_QC = REPO_ROOT / "data/manifests/tcia_kirc/series_geometry_qc.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/embeddings/vision/stunet_turboconv_preliminary"


def release_host_memory() -> None:
    """Return large nnU-Net preprocessing buffers to the OS when possible."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def estimate_case_load(row: pd.Series, dicom_root: Path) -> dict[str, int]:
    series_dir = dicom_root / str(row.case_id) / str(row.SeriesInstanceUID)
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOM slices in {series_dir}")
    dataset = pydicom.dcmread(
        str(files[0]),
        stop_before_pixels=True,
        specific_tags=["Rows", "Columns", "PixelSpacing"],
    )
    spacing_y, spacing_x = map(float, dataset.PixelSpacing)
    target_z = int(round(float(row.ImageCount_num) * float(row.slice_spacing_median_mm) / 1.5))
    target_y = int(round(int(dataset.Rows) * spacing_y / 1.5))
    target_x = int(round(int(dataset.Columns) * spacing_x / 1.5))

    def steps(size: int) -> int:
        return max(1, int(math.ceil((size - 128) / 128.0)) + 1)

    return {
        "target_z": target_z,
        "target_y": target_y,
        "target_x": target_x,
        "estimated_voxels": target_z * target_y * target_x,
        "source_voxels": len(files) * int(dataset.Rows) * int(dataset.Columns),
        "estimated_tiles": steps(target_z) * steps(target_y) * steps(target_x),
    }


def build_or_load_splits(
    geometry_qc_path: Path,
    output_root: Path,
    dicom_root: Path,
    n_calibration: int,
    n_evaluation: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_dir = output_root / "splits"
    calibration_path = split_dir / "calibration.csv"
    evaluation_path = split_dir / "evaluation.csv"
    if calibration_path.exists() and evaluation_path.exists():
        return pd.read_csv(calibration_path), pd.read_csv(evaluation_path)

    qc = pd.read_csv(geometry_qc_path)
    eligible = qc[qc["geometry_qc"].eq("pass")].copy()
    if len(eligible) < n_calibration + n_evaluation:
        raise RuntimeError(
            f"Need {n_calibration+n_evaluation} QC-pass cases, found {len(eligible)}"
        )
    estimates = []
    for _, row in eligible.iterrows():
        estimates.append({**row.to_dict(), **estimate_case_load(row, dicom_root)})
    all_eligible = pd.DataFrame(estimates)
    needed = n_calibration + n_evaluation
    selected_indices = all_eligible.sort_values(
        ["source_voxels", "case_id"], kind="stable"
    ).head(needed).index
    compute_guard_excluded = all_eligible.drop(index=selected_indices).copy()
    compute_guard_excluded["exclusion_reason"] = "largest_source_voxel_count"
    eligible = all_eligible.loc[selected_indices].sort_values(
        ["estimated_voxels", "case_id"], kind="stable"
    ).reset_index(drop=True)

    calibration_indices = np.unique(
        np.rint(np.linspace(0, len(eligible) - 1, n_calibration)).astype(int)
    )
    if len(calibration_indices) != n_calibration:
        raise RuntimeError("Could not construct the requested calibration size")
    calibration = eligible.iloc[calibration_indices].copy()
    remaining = eligible.drop(index=calibration_indices).sort_values(
        ["estimated_voxels", "case_id"], kind="stable"
    )
    evaluation = remaining.iloc[:n_evaluation].copy()
    excluded = pd.concat(
        [remaining.iloc[n_evaluation:].copy(), compute_guard_excluded],
        ignore_index=True,
    )
    if set(calibration.case_id) & set(evaluation.case_id):
        raise AssertionError("Calibration and evaluation cohorts overlap")
    split_dir.mkdir(parents=True, exist_ok=True)
    all_eligible.to_csv(split_dir / "eligible_qc_pass.csv", index=False)
    eligible.to_csv(split_dir / "selected_compute_feasible.csv", index=False)
    calibration.to_csv(calibration_path, index=False)
    evaluation.to_csv(evaluation_path, index=False)
    excluded.to_csv(split_dir / "excluded_compute_guard.csv", index=False)
    return calibration, evaluation


def ensure_nifti(row: pd.Series, dicom_root: Path, inputs_root: Path) -> Path:
    case_id = str(row.case_id)
    nifti_path = inputs_root / case_id / f"{case_id}_0000.nii.gz"
    marker = inputs_root / case_id / "input.json"
    if nifti_path.exists() and marker.exists():
        return nifti_path
    series_dir = dicom_root / case_id / str(row.SeriesInstanceUID)
    metrics = convert_dicom_series(series_dir, str(row.SeriesInstanceUID), nifti_path)
    atomic_json_dump(
        {
            "case_id": case_id,
            "SeriesInstanceUID": str(row.SeriesInstanceUID),
            **metrics,
        },
        marker,
    )
    return nifti_path


def preprocess_without_network(
    nifti_path: Path, model_root: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the official nnU-Net preprocessor without materializing STU-Net."""
    nnunet_root = model_root.resolve() / "STU-Net/nnUNet-1.7.1"
    if str(nnunet_root) not in sys.path:
        sys.path.insert(0, str(nnunet_root))
    os.environ.setdefault("RESULTS_FOLDER", str(model_root.resolve()))
    os.environ.setdefault("nnUNet_raw_data_base", str(model_root.resolve() / "raw"))
    os.environ.setdefault("nnUNet_preprocessed", str(model_root.resolve() / "preprocessed"))
    metadata_path = (
        model_root.resolve()
        / "nnUNet/3d_fullres/Task101_TotalSegmentator"
        / "STUNetTrainer_small__nnUNetPlansv2.1/fold_0/small_ep4k.model.pkl"
    )
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    plans = metadata["plans"]
    from nnunet.training.model_restore import recursive_find_python_class
    import nnunet

    name = plans.get("preprocessor_name") or "GenericPreprocessor"
    preprocessor_class = recursive_find_python_class(
        [str(Path(nnunet.__path__[0]) / "preprocessing")],
        name,
        current_module="nnunet.preprocessing",
    )
    if preprocessor_class is None:
        raise RuntimeError(f"Could not locate nnU-Net preprocessor {name}")
    intensity = plans.get("dataset_properties", {}).get("intensityproperties")
    preprocessor = preprocessor_class(
        plans["normalization_schemes"],
        plans["use_mask_for_norm"],
        plans["transpose_forward"],
        intensity,
    )
    data, _, properties = preprocessor.preprocess_test_case(
        [str(nifti_path)], plans["plans_per_stage"][0]["current_spacing"]
    )
    return data.astype(np.float32, copy=False), properties


def ensure_preprocessed_cache(
    row: pd.Series, args: argparse.Namespace
) -> tuple[Path, Path]:
    case_id = str(row.case_id)
    cache_root = args.output_root / "inputs" / case_id / "preprocessed"
    data_path = cache_root / "data.npy"
    properties_path = cache_root / "properties.pkl"
    marker = cache_root / "complete.json"
    if data_path.exists() and properties_path.exists() and marker.exists():
        return data_path, properties_path
    nifti = ensure_nifti(row, args.dicom_root, args.output_root / "inputs")
    data, properties = preprocess_without_network(nifti, args.model_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    data_temporary = cache_root / "data.tmp.npy"
    properties_temporary = cache_root / "properties.tmp.pkl"
    np.save(data_temporary, data)
    with properties_temporary.open("wb") as handle:
        pickle.dump(properties, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(data_temporary, data_path)
    os.replace(properties_temporary, properties_path)
    atomic_json_dump(
        {
            "case_id": case_id,
            "shape_cxyz": list(map(int, data.shape)),
            "dtype": str(data.dtype),
            "method": "official_nnunet_preprocessor_without_network",
        },
        marker,
    )
    return data_path, properties_path


def centered_patch(data: np.ndarray, patch_size: np.ndarray) -> np.ndarray:
    from batchgenerators.augmentations.utils import pad_nd_image

    padded, _ = pad_nd_image(
        data, patch_size, "constant", {"constant_values": 0}, True, None
    )
    starts = [(padded.shape[i + 1] - int(patch_size[i])) // 2 for i in range(3)]
    return padded[
        None,
        :,
        starts[0] : starts[0] + patch_size[0],
        starts[1] : starts[1] + patch_size[1],
        starts[2] : starts[2] + patch_size[2],
    ]


def low_memory_center_patch(nifti_path: Path, patch_size: np.ndarray) -> np.ndarray:
    """Resample only a physical center patch for a host-memory calibration fallback."""
    image = sitk.ReadImage(str(nifti_path))
    output_size_xyz = [int(value) for value in patch_size[::-1]]
    output_spacing = np.asarray([1.5, 1.5, 1.5], dtype=float)
    center = np.asarray(
        image.TransformContinuousIndexToPhysicalPoint(
            [(value - 1) / 2.0 for value in image.GetSize()]
        )
    )
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    half_extent = output_spacing * (np.asarray(output_size_xyz) - 1) / 2.0
    origin = center - direction @ half_extent
    patch_image = sitk.Resample(
        image,
        output_size_xyz,
        sitk.Transform(),
        sitk.sitkLinear,
        origin.tolist(),
        output_spacing.tolist(),
        image.GetDirection(),
        0.0,
        sitk.sitkFloat32,
    )
    patch = sitk.GetArrayFromImage(patch_image).astype(np.float32, copy=False)
    patch = (patch - float(patch.mean())) / (float(patch.std()) + 1e-8)
    return patch[None, None]


def calibrate_activation_maxabs(
    runtime: STUNetRuntime,
    calibration: pd.DataFrame,
    dicom_root: Path,
    inputs_root: Path,
    output_path: Path,
    rotation_seed: int,
    low_memory: bool = False,
) -> dict[str, dict[str, float]]:
    if output_path.exists():
        return json.loads(output_path.read_text())["layers"]

    layers: dict[str, dict[str, float]] = {}
    hooks = []
    for name, module in runtime.network.named_modules():
        if not isinstance(module, torch.nn.Conv3d):
            continue
        layers[name] = {"ptq_maxabs": 0.0, "turboconv_maxabs": 0.0}
        signs = deterministic_signs(module.in_channels, name, rotation_seed)

        def pre_hook(_module, inputs, layer_name=name, layer_signs=signs):
            values = inputs[0].detach()
            layers[layer_name]["ptq_maxabs"] = max(
                layers[layer_name]["ptq_maxabs"], float(values.abs().max())
            )
            if values.shape[1] > 1 and is_power_of_two(values.shape[1]):
                rotated = rotate_channels(values, layer_signs, dim=1)
                maximum = float(rotated.abs().max())
                del rotated
            else:
                maximum = float(values.abs().max())
            layers[layer_name]["turboconv_maxabs"] = max(
                layers[layer_name]["turboconv_maxabs"], maximum
            )

        hooks.append(module.register_forward_pre_hook(pre_hook))

    started = time.perf_counter()
    try:
        ordered = calibration.sort_values("estimated_voxels", ascending=False)
        for position, (_, row) in enumerate(ordered.iterrows(), 1):
            nifti = ensure_nifti(row, dicom_root, inputs_root)
            if low_memory:
                data = low_memory_center_patch(nifti, runtime.patch_size)
                patch = torch.from_numpy(data).to(runtime.device)
            else:
                data, _ = runtime.preprocess(nifti)
                patch = torch.from_numpy(centered_patch(data, runtime.patch_size)).to(
                    runtime.device
                )
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                _ = runtime.network(patch)
            del data, patch
            release_host_memory()
            torch.cuda.empty_cache()
            print(
                f"[calibration {position}/{len(calibration)}] {row.case_id}",
                flush=True,
            )
    finally:
        for hook in hooks:
            hook.remove()

    payload = {
        "schema_version": 1,
        "n_calibration": len(calibration),
        "case_ids": calibration.case_id.tolist(),
        "rotation_seed": rotation_seed,
        "calibration_preprocessing": (
            "physical_center_patch_fallback" if low_memory else "nnunet_full_volume"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "layers": layers,
    }
    atomic_json_dump(payload, output_path)
    return layers


def activation_scales(
    calibration: dict[str, dict[str, float]], variant: str, bits: int
) -> dict[str, float]:
    qmax = 2 ** (bits - 1) - 1
    key = "ptq_maxabs" if variant == "ptq" else "turboconv_maxabs"
    return {
        name: max(float(values[key]) / qmax, 1e-12)
        for name, values in calibration.items()
    }


def calibrate_activation_maxabs_sharded(
    args: argparse.Namespace,
    calibration: pd.DataFrame,
    output_path: Path,
) -> dict[str, dict[str, float]]:
    """Calibrate in fresh processes so 3D resampling arenas cannot accumulate."""
    if output_path.exists():
        return json.loads(output_path.read_text())["layers"]
    shard_root = output_path.parent / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    ordered = calibration.sort_values("estimated_voxels", ascending=False)
    started = time.perf_counter()
    for position, case_id in enumerate(ordered.case_id.astype(str), 1):
        shard_path = shard_root / f"{case_id}.json"
        if shard_path.exists():
            print(f"[calibration shard {position}/{len(ordered)}] {case_id}: cached")
            continue
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--geometry-qc", str(args.geometry_qc),
            "--dicom-root", str(args.dicom_root),
            "--model-root", str(args.model_root),
            "--output-root", str(args.output_root),
            "--n-calibration", str(args.n_calibration),
            "--n-evaluation", str(args.n_evaluation),
            "--rotation-seed", str(args.rotation_seed),
            "--device", str(args.device),
            "--precision", str(args.precision),
            "--step-size", str(args.step_size),
            "--calibration-shard-case", case_id,
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    aggregate: dict[str, dict[str, float]] = {}
    for case_id in calibration.case_id.astype(str):
        payload = json.loads((shard_root / f"{case_id}.json").read_text())
        for name, values in payload["layers"].items():
            current = aggregate.setdefault(
                name, {"ptq_maxabs": 0.0, "turboconv_maxabs": 0.0}
            )
            for key in current:
                current[key] = max(current[key], float(values[key]))
    atomic_json_dump(
        {
            "schema_version": 1,
            "method": "max_of_independent_patient_shards",
            "n_calibration": len(calibration),
            "case_ids": calibration.case_id.astype(str).tolist(),
            "rotation_seed": args.rotation_seed,
            "elapsed_seconds": time.perf_counter() - started,
            "layers": aggregate,
        },
        output_path,
    )
    return aggregate


def original_conv_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Conv3d)
    }


def run_weight_sanity(
    runtime: STUNetRuntime,
    calibration: dict[str, dict[str, float]],
    output_path: Path,
    rotation_seed: int,
) -> pd.DataFrame:
    if output_path.exists():
        return pd.read_csv(output_path)
    rows = []
    original = original_conv_weights(runtime.network)
    for bits in (8, 6, 4):
        for variant in ("ptq", "turboconv"):
            modules: dict[str, FakeQuantConv3d] = {}
            spec = QuantizationSpec(variant, bits, 8, rotation_seed)
            scales = activation_scales(calibration, variant, 8)
            for name, module in runtime.network.named_modules():
                if isinstance(module, torch.nn.Conv3d):
                    modules[name] = FakeQuantConv3d(
                        module, name, spec, scales[name], quantize=True
                    )
            rows.append(
                {
                    "variant": variant,
                    "weight_bits": bits,
                    "activation_bits": 8,
                    **weight_error_summary(original, modules),
                }
            )
            del modules
            gc.collect()
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def variant_definition(name: str, weight_bits: int, activation_bits: int) -> dict[str, Any]:
    if name == "fp32":
        return {"name": name, "quantized": False}
    return {
        "name": name,
        "quantized": True,
        "spec": QuantizationSpec(name, weight_bits, activation_bits),
    }


def run_variant(
    name: str,
    evaluation: pd.DataFrame,
    calibration: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> None:
    variant_root = args.output_root / "variants" / name
    selected_ids = set(evaluation.case_id.astype(str))
    complete = {
        path.parent.name
        for path in (variant_root / "cases").glob("*/complete.json")
        if (path.parent / f"{path.parent.name}_stunet_seg.nii.gz").exists()
        and (path.parent / "logit_sketch.npz").exists()
    }
    if complete == selected_ids:
        print(f"[{name}] already complete ({len(complete)})", flush=True)
        return

    runtime = STUNetRuntime(
        args.model_root, args.device, args.step_size, args.precision
    )
    weight_metrics: dict[str, float] | None = None
    model_quantized = name == "fp32"

    checkpoint_hash = sha256_file(runtime.checkpoint_path)
    ordered = evaluation.sort_values("estimated_voxels", ascending=False)
    for position, (_, row) in enumerate(ordered.iterrows(), 1):
        case_id = str(row.case_id)
        case_root = variant_root / "cases" / case_id
        marker = case_root / "complete.json"
        failure_marker = case_root / "failure.json"
        segmentation_path = case_root / f"{case_id}_stunet_seg.nii.gz"
        logit_path = case_root / "logit_sketch.npz"
        if marker.exists() and segmentation_path.exists() and logit_path.exists():
            print(f"[{name} {position}/{len(evaluation)}] {case_id}: cached", flush=True)
            continue
        work_dir = case_root / "work"
        shutil.rmtree(work_dir, ignore_errors=True)
        try:
            nifti = ensure_nifti(row, args.dicom_root, args.output_root / "inputs")
            with PeakRSSMonitor() as memory:
                prepared = None
                if not model_quantized:
                    cache_root = args.output_root / "inputs" / case_id / "preprocessed"
                    data_path = cache_root / "data.npy"
                    properties_path = cache_root / "properties.pkl"
                    if data_path.exists() and properties_path.exists():
                        # Copy-on-write keeps disk-backed pages while presenting a
                        # writable view accepted by torch.from_numpy.
                        data = np.load(data_path, mmap_mode="c")
                        with properties_path.open("rb") as handle:
                            properties = pickle.load(handle)
                        prepared = (data, properties)
                    else:
                        prepared = runtime.preprocess(nifti)
                    original = original_conv_weights(runtime.network)
                    spec = QuantizationSpec(
                        name, args.weight_bits, args.activation_bits, args.rotation_seed
                    )
                    modules = replace_conv3d_modules(
                        runtime.network,
                        spec,
                        activation_scales(calibration, name, args.activation_bits),
                    )
                    weight_metrics = weight_error_summary(original, modules)
                    runtime.network.eval()
                    del original, modules
                    release_host_memory()
                    torch.cuda.empty_cache()
                    model_quantized = True
                embedding, metrics = runtime.run_case(
                    nifti,
                    segmentation_path,
                    work_dir,
                    args.roi_margin_mm,
                    logit_sketch_path=logit_path,
                    export_order=args.export_order,
                    preprocessed=prepared,
                )
            metrics["peak_rss_gib"] = memory.peak_bytes / 2**30
            payload = {
                "schema_version": 1,
                "case_id": case_id,
                "SeriesInstanceUID": str(row.SeriesInstanceUID),
                "variant": name,
                "weight_bits": 32 if name == "fp32" else args.weight_bits,
                "activation_bits": 16 if name == "fp32" else args.activation_bits,
                "checkpoint_sha256": checkpoint_hash,
                "embedding": embedding.astype(float).tolist(),
                "metrics": metrics,
                "weight_metrics": weight_metrics,
            }
            atomic_json_dump(payload, marker)
            failure_marker.unlink(missing_ok=True)
            shutil.rmtree(work_dir, ignore_errors=True)
            release_host_memory()
            print(
                f"[{name} {position}/{len(evaluation)}] {case_id}: "
                f"{metrics['total_seconds']:.1f}s tiles={metrics['n_tiles']}",
                flush=True,
            )
        except Exception as exc:
            shutil.rmtree(work_dir, ignore_errors=True)
            atomic_json_dump(
                {
                    "case_id": case_id,
                    "variant": name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                failure_marker,
            )
            print(f"[{name} {position}] {case_id}: FAILED {exc!r}", flush=True)
    del runtime
    gc.collect()
    torch.cuda.empty_cache()


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    denominator = int(mask_a.sum() + mask_b.sum())
    return 1.0 if denominator == 0 else 2.0 * float(np.logical_and(mask_a, mask_b).sum()) / denominator


def analyze_results(
    evaluation: pd.DataFrame,
    output_root: Path,
    variants: tuple[str, ...] = ("ptq", "turboconv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_rows = []
    for case_id in evaluation.case_id.astype(str):
        base_root = output_root / "variants/fp32/cases" / case_id
        base = json.loads((base_root / "complete.json").read_text())
        base_embedding = np.asarray(base["embedding"], dtype=np.float64)
        base_mask = sitk.GetArrayFromImage(
            sitk.ReadImage(str(base_root / f"{case_id}_stunet_seg.nii.gz"))
        )
        base_logits = np.load(base_root / "logit_sketch.npz")["logits"].astype(np.float32)
        for variant in variants:
            current_root = output_root / "variants" / variant / "cases" / case_id
            marker = current_root / "complete.json"
            if not marker.exists():
                continue
            current = json.loads(marker.read_text())
            embedding = np.asarray(current["embedding"], dtype=np.float64)
            mask = sitk.GetArrayFromImage(
                sitk.ReadImage(str(current_root / f"{case_id}_stunet_seg.nii.gz"))
            )
            logits = np.load(current_root / "logit_sketch.npz")["logits"].astype(np.float32)
            if logits.shape != base_logits.shape:
                raise ValueError(f"Logit sketch shape mismatch for {case_id} {variant}")
            logit_difference = logits - base_logits
            logit_l2 = np.linalg.norm(logit_difference.ravel())
            base_l2 = max(np.linalg.norm(base_logits.ravel()), 1e-12)
            embedding_cosine = float(
                np.dot(base_embedding, embedding)
                / max(np.linalg.norm(base_embedding) * np.linalg.norm(embedding), 1e-12)
            )
            base_kidney = (base_mask == 38) | (base_mask == 39)
            kidney = (mask == 38) | (mask == 39)
            paired_rows.append(
                {
                    "case_id": case_id,
                    "variant": variant,
                    "embedding_cosine": embedding_cosine,
                    "embedding_rmse": float(np.sqrt(np.mean((embedding - base_embedding) ** 2))),
                    "embedding_max_abs": float(np.max(np.abs(embedding - base_embedding))),
                    "logit_rmse": float(np.sqrt(np.mean(logit_difference**2))),
                    "logit_relative_l2": float(logit_l2 / base_l2),
                    "logit_cosine": float(
                        np.dot(logits.ravel(), base_logits.ravel())
                        / max(np.linalg.norm(logits.ravel()) * base_l2, 1e-12)
                    ),
                    "mask_dice_kidney_union": dice(base_kidney, kidney),
                    "mask_dice_left": dice(base_mask == 38, mask == 38),
                    "mask_dice_right": dice(base_mask == 39, mask == 39),
                    "mask_global_agreement": float(np.mean(base_mask == mask)),
                    "kidney_volume_delta_pct": float(
                        100.0 * (kidney.sum() - base_kidney.sum()) / max(base_kidney.sum(), 1)
                    ),
                    "runtime_seconds": float(current["metrics"]["total_seconds"]),
                    "gpu_peak_allocated_gib": float(
                        current["metrics"]["gpu_peak_allocated_gib"]
                    ),
                }
            )
    paired = pd.DataFrame(paired_rows)
    summaries = []
    metrics = [
        "embedding_cosine",
        "embedding_rmse",
        "logit_rmse",
        "logit_relative_l2",
        "logit_cosine",
        "mask_dice_kidney_union",
        "mask_dice_left",
        "mask_dice_right",
        "mask_global_agreement",
        "kidney_volume_delta_pct",
        "runtime_seconds",
        "gpu_peak_allocated_gib",
    ]
    for variant, group in paired.groupby("variant"):
        row: dict[str, Any] = {"variant": variant, "n_cases": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_p05"] = float(np.percentile(values, 5))
            row[f"{metric}_p95"] = float(np.percentile(values, 95))
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    paired.to_csv(output_root / "paired_drift.csv", index=False)
    summary.to_csv(output_root / "summary.csv", index=False)
    write_quantization_verdict(paired, summary, output_root)
    return paired, summary


def write_quantization_verdict(
    paired: pd.DataFrame, summary: pd.DataFrame, output_root: Path
) -> dict[str, Any]:
    """Apply predeclared W4A8 retention gates and a paired PTQ comparison."""
    gates = {
        "embedding_cosine_median_min": 0.99,
        "logit_cosine_median_min": 0.99,
        "mask_dice_kidney_union_median_min": 0.95,
        "mask_dice_kidney_union_p05_min": 0.90,
    }
    retention: dict[str, dict[str, Any]] = {}
    for _, row in summary.iterrows():
        variant = str(row["variant"])
        checks = {
            name: bool(float(row[name.removesuffix("_min")]) >= threshold)
            for name, threshold in gates.items()
        }
        retention[variant] = {"passes": all(checks.values()), "checks": checks}

    required_variants = {"ptq", "turboconv"}
    present_variants = set(paired["variant"].astype(str).unique())
    missing_variants = sorted(required_variants - present_variants)
    if missing_variants:
        payload = {
            "schema_version": 1,
            "scope": "W4A8 fake-quant numerical kill test",
            "analysis_complete": False,
            "missing_variants": missing_variants,
            "retention_gates": gates,
            "retention": retention,
            "paired_comparison": {},
            "recommendation": "insufficient_paired_variants",
            "latency_conclusion_allowed": False,
            "reason_latency_not_conclusive": (
                "Both variants use float fake-quant kernels; TurboConv includes an "
                "unfused Python/PyTorch Hadamard transform."
            ),
            "theoretical_weight_storage_reduction_vs_fp32": 8.0,
        }
        atomic_json_dump(payload, output_root / "verdict.json")
        return payload

    wide = paired.pivot(index="case_id", columns="variant")
    comparisons = {}
    for metric, higher_is_better in {
        "embedding_cosine": True,
        "logit_relative_l2": False,
        "mask_dice_kidney_union": True,
    }.items():
        turbo = wide[(metric, "turboconv")]
        ptq = wide[(metric, "ptq")]
        delta = turbo - ptq
        comparisons[metric] = {
            "turbo_minus_ptq_median": float(delta.median()),
            "turbo_win_fraction": float((delta > 0).mean())
            if higher_is_better
            else float((delta < 0).mean()),
        }

    turbo_passes = retention.get("turboconv", {}).get("passes", False)
    ptq_passes = retention.get("ptq", {}).get("passes", False)
    no_material_regression = (
        comparisons["embedding_cosine"]["turbo_minus_ptq_median"] >= -0.002
        and comparisons["mask_dice_kidney_union"]["turbo_minus_ptq_median"] >= -0.002
    )
    improves_logits = (
        comparisons["logit_relative_l2"]["turbo_minus_ptq_median"] < 0
    )
    if turbo_passes and improves_logits and no_material_regression:
        recommendation = "turboconv_numerically_preferred"
    elif ptq_passes:
        recommendation = "ptq_preferred_for_simplicity"
    elif turbo_passes:
        recommendation = "turboconv_viable_but_not_superior_to_ptq"
    else:
        recommendation = "reject_w4a8_for_both_variants"
    payload = {
        "schema_version": 1,
        "scope": "W4A8 fake-quant numerical kill test",
        "analysis_complete": True,
        "missing_variants": [],
        "retention_gates": gates,
        "retention": retention,
        "paired_comparison": comparisons,
        "recommendation": recommendation,
        "latency_conclusion_allowed": False,
        "reason_latency_not_conclusive": (
            "Both variants use float fake-quant kernels; TurboConv includes an "
            "unfused Python/PyTorch Hadamard transform."
        ),
        "theoretical_weight_storage_reduction_vs_fp32": 8.0,
    }
    atomic_json_dump(payload, output_root / "verdict.json")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-qc", type=Path, default=DEFAULT_GEOMETRY_QC)
    parser.add_argument("--series-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dicom-root", type=Path, default=DEFAULT_DICOM_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-calibration", type=int, default=25)
    parser.add_argument("--n-evaluation", type=int, default=50)
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--rotation-seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["amp", "fp32"], default="amp")
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--roi-margin-mm", type=float, default=30.0)
    parser.add_argument(
        "--variants", nargs="+", choices=["fp32", "ptq", "turboconv"],
        default=["fp32", "ptq", "turboconv"],
    )
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--calibration-shard-case", type=str)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--calibration-low-memory", action="store_true")
    parser.add_argument("--evaluation-shard-case", type=str)
    parser.add_argument("--export-order", type=int, choices=[0, 1], default=1)
    parser.add_argument("--preprocess-shard-case", type=str)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    calibration_cases, evaluation_cases = build_or_load_splits(
        args.geometry_qc,
        args.output_root,
        args.dicom_root,
        args.n_calibration,
        args.n_evaluation,
    )
    print(
        f"Frozen cohorts: calibration={len(calibration_cases)}, "
        f"evaluation={len(evaluation_cases)}, overlap="
        f"{len(set(calibration_cases.case_id)&set(evaluation_cases.case_id))}",
        flush=True,
    )
    if args.prepare_only:
        return 0
    if args.preprocess_shard_case:
        selected = evaluation_cases[
            evaluation_cases.case_id.astype(str).eq(args.preprocess_shard_case)
        ]
        if len(selected) != 1:
            raise RuntimeError(
                f"Evaluation case not found uniquely: {args.preprocess_shard_case}"
            )
        ensure_preprocessed_cache(selected.iloc[0], args)
        print(f"[preprocessed] {args.preprocess_shard_case}", flush=True)
        return 0
    if args.calibration_shard_case:
        selected = calibration_cases[
            calibration_cases.case_id.astype(str).eq(args.calibration_shard_case)
        ]
        if len(selected) != 1:
            raise RuntimeError(
                f"Calibration case not found uniquely: {args.calibration_shard_case}"
            )
        runtime = STUNetRuntime(
            args.model_root, args.device, args.step_size, args.precision
        )
        calibrate_activation_maxabs(
            runtime,
            selected,
            args.dicom_root,
            args.output_root / "inputs",
            args.output_root / "calibration/shards"
            / f"{args.calibration_shard_case}.json",
            args.rotation_seed,
            low_memory=args.calibration_low_memory,
        )
        return 0
    if args.analyze_only:
        _, summary = analyze_results(evaluation_cases, args.output_root)
        print(summary.to_string(index=False), flush=True)
        return 0

    calibration_path = args.output_root / "calibration/activation_maxabs.json"
    if args.evaluation_shard_case:
        if len(args.variants) != 1:
            raise RuntimeError("Evaluation shards require exactly one --variants value")
        selected = evaluation_cases[
            evaluation_cases.case_id.astype(str).eq(args.evaluation_shard_case)
        ]
        if len(selected) != 1:
            raise RuntimeError(
                f"Evaluation case not found uniquely: {args.evaluation_shard_case}"
            )
        if not calibration_path.exists():
            raise RuntimeError("Aggregated calibration is required before evaluation")
        calibration = json.loads(calibration_path.read_text())["layers"]
        run_variant(args.variants[0], selected, calibration, args)
        marker = (
            args.output_root / "variants" / args.variants[0] / "cases"
            / args.evaluation_shard_case / "complete.json"
        )
        return 0 if marker.exists() else 1
    calibration = calibrate_activation_maxabs_sharded(
        args,
        calibration_cases,
        calibration_path,
    )
    runtime = STUNetRuntime(
        args.model_root, args.device, args.step_size, args.precision
    )
    weight_sanity = run_weight_sanity(
        runtime,
        calibration,
        args.output_root / "calibration/weight_sanity.csv",
        args.rotation_seed,
    )
    print("Weight sanity:\n" + weight_sanity.to_string(index=False), flush=True)
    del runtime
    release_host_memory()
    torch.cuda.empty_cache()

    for variant in args.variants:
        run_variant(variant, evaluation_cases, calibration, args)

    requested_complete = all(
        sum(
            1
            for case_id in evaluation_cases.case_id.astype(str)
            if (
                args.output_root
                / "variants"
                / variant
                / "cases"
                / case_id
                / "complete.json"
            ).exists()
        )
        == len(evaluation_cases)
        for variant in args.variants
    )
    all_complete = requested_complete and set(args.variants) == {
        "fp32", "ptq", "turboconv"
    }
    if all_complete:
        _, summary = analyze_results(evaluation_cases, args.output_root)
        print("Summary:\n" + summary.to_string(index=False), flush=True)
    provenance = {
        "schema_version": 1,
        "n_calibration": len(calibration_cases),
        "n_evaluation": len(evaluation_cases),
        "calibration_evaluation_overlap": len(
            set(calibration_cases.case_id) & set(evaluation_cases.case_id)
        ),
        "variants": args.variants,
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "rotation_seed": args.rotation_seed,
        "precision_reference": args.precision,
        "fake_quantization": True,
        "optimized_integer_kernel": False,
        "checkpoint_sha256": sha256_file(
            args.model_root
            / "nnUNet/3d_fullres/Task101_TotalSegmentator"
            / "STUNetTrainer_small__nnUNetPlansv2.1/fold_0/small_ep4k.model"
        ),
    }
    atomic_json_dump(provenance, args.output_root / "provenance.json")
    return 0 if requested_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
