"""Verify source hashes and paired cache geometry; write local QC montage."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from build_renal_2p5d_program_cache import sha, aligned_arrays, ARMS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    done_files = sorted((args.cache / 'images').glob('*.json'))
    selected = set(np.linspace(0, len(done_files)-1, 4, dtype=int))
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ratios, single, errors = [], 0, []
    for index, f in enumerate(done_files):
        d = json.loads(f.read_text())
        case = d['case_id']
        root = args.source / 'cases' / case
        image = root / 'input' / f'{case}_0000.nii.gz'
        mask = root / f'{case}_stunet_seg.nii.gz'
        for path, key in [(image, 'input_sha256'), (mask, 'mask_sha256'), (root/'complete.json', 'source_marker_sha256')]:
            if sha(path) != d[key]:
                errors.append('Source hash mismatch')
        positions = {}
        for arm in ARMS:
            with np.load(args.cache / arm / 'cases' / f'{case}.npz') as data:
                if not np.isfinite(data['features']).all() or data['features'].shape[1] != 512:
                    errors.append('Invalid features')
                positions[arm] = data['center_indices']
        if not np.array_equal(positions['renal_slices'], positions['renal_crop']):
            errors.append('Crop/control centers differ')
        (z0,z1), (y0,y1), (x0,x1) = d['box_zyx']
        ratios.append((y1-y0)*(x1-x0)/(d['shape_zyx'][1]*d['shape_zyx'][2]))
        single += int(d['radiomics'][-1])
        if index in selected:
            vol, seg, _ = aligned_arrays(image, mask)
            renal = np.isin(seg, [38,39])
            z = int(renal.sum(axis=(1,2)).argmax())
            ax = axes.flat[sorted(selected).index(index)]
            ax.imshow(vol[z], cmap='gray', vmin=-150, vmax=250)
            ax.contour(renal[z], levels=[.5], colors=['lime'], linewidths=.5)
            ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, edgecolor='orange'))
            ax.set_title(f'QC sample {sorted(selected).index(index)+1}; axial {z}; renal crop')
            ax.axis('off')
    fig.tight_layout()
    # Patient-derived montage remains under ignored data, not public results.
    fig.savefig(args.cache / 'geometry_qc.png', dpi=130)
    plt.close(fig)
    result = {'cases': len(done_files), 'errors': errors, 'single_kidney_cases': single,
              'crop_fraction_xy_min_median_max': np.quantile(ratios, [0,.5,1]).tolist(),
              'source_hashes_verified': not errors, 'paired_axial_centers_verified': not errors,
              'scope': 'Geometry/hash verification plus four outcome-independent QC samples; not expert tumor segmentation validation'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if errors:
        raise RuntimeError('Cache audit failed')


if __name__ == '__main__':
    main()
