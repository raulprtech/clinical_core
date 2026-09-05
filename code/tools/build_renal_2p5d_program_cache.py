"""Outcome-independent renal crops, frozen sequences and explicit 2D features.

Requires real, geometrically aligned STU-Net masks; never synthesizes masks or
features. All arms read the same converted CT. Source caches stay immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from skimage.measure import find_contours

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.adapters.ingestion.vision.models.resnet_multiview import VisionResNet18_2p5D

ARMS = ('full', 'renal_slices', 'renal_crop')
FEATURES = ('mean_hu', 'std_hu', 'p10_hu', 'median_hu', 'p90_hu', 'iqr_hu',
            'entropy16', 'area_mm2', 'perimeter_mm', 'compactness',
            'glcm_contrast', 'glcm_homogeneity', 'glcm_energy')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def aligned_arrays(image_path, mask_path):
    image, segmentation = sitk.ReadImage(str(image_path)), sitk.ReadImage(str(mask_path))
    for prop in ('Size', 'Spacing', 'Origin', 'Direction'):
        if not np.allclose(getattr(image, 'Get' + prop)(),
                           getattr(segmentation, 'Get' + prop)(), rtol=0, atol=1e-4):
            raise ValueError('Image/mask geometry mismatch: ' + prop)
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    mask = sitk.GetArrayFromImage(segmentation)
    if not np.isfinite(volume).all():
        raise ValueError('Nonfinite CT values')
    return volume, mask, np.array(image.GetSpacing())[::-1]


def renal_box(labels, spacing, margin_mm=10.):
    points = np.where(np.isin(labels, [38, 39]))
    if not len(points[0]):
        raise ValueError('No renal labels in real segmentation')
    margin = np.ceil(margin_mm / np.asarray(spacing)).astype(int)
    return tuple(slice(max(0, int(p.min()) - int(m)),
                       min(n, int(p.max()) + 1 + int(m)))
                 for p, m, n in zip(points, margin, labels.shape))


def plane_features(plane, mask, spacing_yx):
    """13 explicit features; pair counts exclude all pixels outside the ROI."""
    if mask.sum() < 4:
        raise ValueError('Insufficient renal pixels')
    windowed = np.clip(plane, -150., 250.)
    roi = windowed[mask]
    q = np.minimum(15, ((windowed + 150.) / 25.).astype(int))
    hist = np.bincount(q[mask], minlength=16).astype(float)
    hist /= hist.sum()
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    glcm = np.zeros((16, 16), float)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ya = slice(0, plane.shape[0] - dy)
        yb = slice(dy, plane.shape[0])
        xa = slice(max(0, -dx), min(plane.shape[1], plane.shape[1] - dx))
        xb = slice(max(0, dx), min(plane.shape[1], plane.shape[1] + dx))
        valid = mask[ya, xa] & mask[yb, xb]
        a, b = q[ya, xa][valid], q[yb, xb][valid]
        counts = np.bincount(a * 16 + b, minlength=256).reshape(16, 16)
        glcm += counts + counts.T
    if not glcm.sum():
        raise ValueError('No within-ROI texture pairs')
    glcm /= glcm.sum()
    ii, jj = np.indices(glcm.shape)
    delta = (ii - jj) ** 2
    perimeter = sum(np.linalg.norm(np.diff(c, axis=0) * spacing_yx, axis=1).sum()
                    for c in find_contours(np.pad(mask.astype(float), 1), .5))
    area = float(mask.sum() * np.prod(spacing_yx))
    p10, p25, p50, p75, p90 = np.percentile(roi, [10, 25, 50, 75, 90])
    return np.array([roi.mean(), roi.std(), p10, p50, p90, p75 - p25,
                     entropy, area, perimeter, 4 * np.pi * area / max(perimeter**2, 1e-12),
                     np.sum(glcm * delta), np.sum(glcm / (1 + delta)),
                     np.sum(glcm**2)], dtype=float)


def radiomics_2d(volume, labels, spacing):
    sides = []
    for label in (38, 39):
        areas = (labels == label).sum(axis=(1, 2))
        if areas.max() < 4:
            continue
        z = int(areas.argmax())
        sides.append(plane_features(volume[z], labels[z] == label, spacing[1:]))
    if not sides:
        raise ValueError('Neither kidney supports 2D features')
    mean = np.mean(sides, axis=0)
    # A missing second kidney gives no measured asymmetry; absence is explicit.
    difference = np.abs(sides[0] - sides[1]) if len(sides) == 2 else np.zeros_like(mean)
    return np.r_[mean, difference, float(len(sides) == 1)]


def images_at(volume, centers, crop):
    images = []
    for center in centers:
        channels = np.stack([volume[np.clip(center + d, 0, len(volume) - 1), crop[1], crop[2]]
                             for d in (-1, 0, 1)])
        channels = (np.clip(channels, -150., 250.) + 150.) / 400.
        images.append(F.interpolate(torch.from_numpy(channels)[None], (224, 224),
                                    mode='bilinear', align_corners=False)[0])
    return torch.stack(images)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.hub.set_dir('data/models/torch')
    model = VisionResNet18_2p5D(device=args.device, weights_dir='data/models/torch')
    backbone = model._get_backbone().to(args.device).eval()
    feature_names = [f'{kind}_{name}' for kind in ('mean', 'absdiff') for name in FEATURES] + ['single_kidney']
    contract = {'schema': 1, 'source': str(args.source.resolve()), 'margin_mm': 10,
                'max_tokens': 64, 'arms': ARMS, 'radiomics_feature_names': feature_names,
                'encoder': 'ResNet18 ImageNet1K V1 frozen', 'window_hu': [-150, 250],
                'script_sha256': sha(__file__), 'outcome_independent': True}
    args.output.mkdir(parents=True, exist_ok=True)
    contract_path = args.output / 'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != json.loads(json.dumps(contract)):
        raise ValueError('Output has a different extraction contract; use a new directory')
    contract_path.write_text(json.dumps(contract, indent=2) + '\n')
    for arm in ARMS:
        (args.output / arm / 'cases').mkdir(parents=True, exist_ok=True)
    (args.output / 'images').mkdir(exist_ok=True)
    markers = sorted((args.source / 'cases').glob('*/complete.json'))
    if args.limit:
        markers = markers[:args.limit]
    rows, radiomics, failures = [], [], []
    for index, marker in enumerate(markers):
        case = marker.parent.name
        try:
            meta = json.loads(marker.read_text())
            image_path = marker.parent / 'input' / f'{case}_0000.nii.gz'
            mask_path = marker.parent / f'{case}_stunet_seg.nii.gz'
            done_path = args.output / 'images' / f'{case}.json'
            if done_path.exists():
                done = json.loads(done_path.read_text())
                if done['source_marker_sha256'] != sha(marker):
                    raise ValueError('Source marker changed')
                if not all((args.output / arm / 'cases' / f'{case}.npz').exists() for arm in ARMS):
                    raise ValueError('Incomplete completed cache')
            else:
                volume, labels, spacing = aligned_arrays(image_path, mask_path)
                box = renal_box(labels, spacing)
                centers_full = model._uniform_indices(len(volume), 64)
                centers_renal = model._uniform_indices(box[0].stop - box[0].start, 64) + box[0].start
                stats = radiomics_2d(volume, labels, spacing)
                for arm in ARMS:
                    centers = centers_full if arm == 'full' else centers_renal
                    crop = box if arm == 'renal_crop' else tuple(slice(0, n) for n in volume.shape)
                    images = images_at(volume, centers, crop)
                    normalized = (images - model.imagenet_mean) / model.imagenet_std
                    with torch.inference_mode():
                        features = torch.cat([backbone(chunk.to(args.device)).flatten(1).cpu()
                                              for chunk in normalized.split(16)])
                    features = F.normalize(features, dim=1).numpy().astype(np.float16)
                    out = args.output / arm / 'cases' / f'{case}.npz'
                    np.savez_compressed(out, features=features,
                                        positions=(centers / max(1, len(volume)-1)).astype(np.float32),
                                        center_indices=centers)
                    if arm == 'renal_crop':
                        # 16 fixed tokens for the planned last-block adaptation experiment.
                        selected = np.linspace(0, len(images)-1, min(16, len(images)), dtype=int)
                        np.savez_compressed(args.output / 'images' / f'{case}.npz',
                                            images=images[selected].numpy().astype(np.float16),
                                            center_indices=centers[selected])
                done = {'case_id': case, 'SeriesInstanceUID': meta['SeriesInstanceUID'],
                        'source_marker_sha256': sha(marker), 'input_sha256': sha(image_path),
                        'mask_sha256': sha(mask_path), 'shape_zyx': list(volume.shape),
                        'spacing_zyx': spacing.tolist(), 'box_zyx': [[s.start, s.stop] for s in box],
                        'n_full': len(centers_full), 'n_renal': len(centers_renal),
                        'radiomics': stats.tolist()}
                done_path.write_text(json.dumps(done, indent=2) + '\n')
            rows.append({k: v for k, v in done.items() if k != 'radiomics'})
            radiomics.append({'case_id': case, **dict(zip(feature_names, done['radiomics']))})
        except Exception as exc:
            failures.append({'case_id': case, 'error': str(exc)})
        print(f'{index+1}/{len(markers)} valid={len(rows)} failed={len(failures)}', flush=True)
    pd.DataFrame(rows).to_csv(args.output / 'geometry_audit.csv', index=False)
    pd.DataFrame(radiomics).to_csv(args.output / 'radiomics_2d.csv', index=False)
    pd.DataFrame(failures, columns=['case_id', 'error']).to_csv(args.output / 'failures.csv', index=False)
    for arm in ARMS:
        pd.DataFrame([{'case_id': r['case_id'], 'SeriesInstanceUID': r['SeriesInstanceUID'],
                       'sequence_path': str((args.output / arm / 'cases' / (r['case_id']+'.npz')).resolve())}
                      for r in rows]).to_csv(args.output / arm / 'manifest.csv', index=False)
    print(json.dumps({'valid': len(rows), 'failures': len(failures)}))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
