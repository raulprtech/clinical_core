"""Build outcome-independent full-field images for the matched F6 adaptation.

Reuse the exact full-field token centers and CT from the audited renal program.
No labels or survival outcomes are read. Existing caches stay immutable.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
import build_renal_2p5d_program_cache as parent


def selected_centers(centers):
    centers = np.asarray(centers)
    if centers.ndim != 1 or len(centers) == 0:
        raise ValueError('Expected a nonempty vector of token centers')
    return centers[np.linspace(0, len(centers)-1, min(16, len(centers)), dtype=int)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prior = json.loads((args.parent_cache/'contract.json').read_text())
    source = Path(prior['source'])
    contract = {'schema':1, 'arm':'full', 'n_tokens':16, 'outcome_independent':True,
                'parent_contract_sha256':parent.sha(args.parent_cache/'contract.json'),
                'script_sha256':parent.sha(__file__), 'image_helper_sha256':parent.sha(parent.__file__),
                'window_hu':[-150,250], 'neighbors':[-1,0,1], 'size':224,
                'selection':'16 evenly spaced positions from existing full token centers'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'images').mkdir(exist_ok=True)
    path = args.output/'contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Different cache contract; choose a new output directory')
    path.write_text(json.dumps(contract, indent=2)+'\n')
    files = sorted((args.parent_cache/'images').glob('*.json'))
    if not files:
        raise ValueError('No audited parent image metadata')
    completed = 0
    for number, metadata in enumerate(files):
        case = metadata.stem
        meta = json.loads(metadata.read_text())
        image = source/'cases'/case/'input'/f'{case}_0000.nii.gz'
        tokens = args.parent_cache/'full'/'cases'/f'{case}.npz'
        image_hash = parent.sha(image)
        if image_hash != meta['input_sha256']:
            raise ValueError('Audited input CT changed')
        with np.load(tokens, allow_pickle=False) as d:
            centers = selected_centers(d['center_indices'])
        provenance = {'input_sha256':image_hash, 'full_tokens_sha256':parent.sha(tokens),
                      'parent_metadata_sha256':parent.sha(metadata), 'center_indices':centers.tolist()}
        dest = args.output/'images'/f'{case}.npz'
        marker = dest.with_suffix('.json')
        if marker.exists():
            if json.loads(marker.read_text()) != provenance or not dest.exists():
                raise ValueError('Existing image provenance differs or image missing')
        else:
            volume = sitk.GetArrayFromImage(sitk.ReadImage(str(image))).astype(np.float32)
            if not np.isfinite(volume).all() or list(volume.shape) != meta['shape_zyx']:
                raise ValueError('CT shape or intensity audit failed')
            if centers.min()<0 or centers.max()>=len(volume):
                raise ValueError('Full token center out of CT bounds')
            crop = tuple(slice(0, n) for n in volume.shape)
            images = parent.images_at(volume, centers, crop).numpy().astype(np.float16)
            np.savez_compressed(dest, images=images, center_indices=centers)
            marker.write_text(json.dumps(provenance, indent=2)+'\n')
        with np.load(dest, allow_pickle=False) as d:
            assert d['images'].shape == (len(centers),3,224,224)
            assert np.isfinite(d['images']).all()
            assert d['images'].min()>=0 and d['images'].max()<=1
            np.testing.assert_array_equal(d['center_indices'],centers)
        completed += 1
        print(f'full-field images {number+1}/{len(files)} verified', flush=True)
    audit = {'cases':completed,'source_hashes_verified':True,'full_centers_verified':True,
             'images_finite_in_unit_interval':True,'n_tokens':16,'outcome_independent':True,
             'visual_review_performed':False}
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
