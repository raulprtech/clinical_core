"""Outcome-independent F7 renal crops preserving pixel aspect ratio."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from build_renal_2p5d_program_cache import sha, VisionResNet18_2p5D


def square_pad(channels):
    h,w = channels.shape[-2:]
    side = max(h,w)
    left,top = (side-w)//2,(side-h)//2
    return F.pad(channels,(left,side-w-left,top,side-h-top),value=0.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.hub.set_dir('data/models/torch')
    old=json.loads((args.parent_cache/'contract.json').read_text())
    source=Path(old['source'])
    contract={'schema':1,'arm':'renal_letterbox','parent_contract_sha256':sha(args.parent_cache/'contract.json'),
              'script_sha256':sha(__file__),'outcome_independent':True,'padding':'symmetric square at -150 HU',
              'window_hu':[-150,250],'neighbors':[-1,0,1],'max_tokens':64,'encoder':'ResNet18 ImageNet1K V1 frozen'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'cases').mkdir(exist_ok=True)
    path=args.output/'contract.json'
    if path.exists() and json.loads(path.read_text())!=contract:
        raise ValueError('Extraction contract changed')
    path.write_text(json.dumps(contract,indent=2)+'\n')
    encoder=VisionResNet18_2p5D(device=args.device,weights_dir='data/models/torch')
    backbone=encoder._get_backbone().to(args.device).eval()
    files=sorted((args.parent_cache/'images').glob('*.json'))
    if not files:
        raise ValueError('No parent cases')
    manifest,ratios=[],[]
    for number,metadata in enumerate(files):
        case=metadata.stem
        meta=json.loads(metadata.read_text())
        image=source/'cases'/case/'input'/f'{case}_0000.nii.gz'
        original=args.parent_cache/'renal_crop'/'cases'/f'{case}.npz'
        if sha(image)!=meta['input_sha256']:
            raise ValueError('Audited CT changed')
        with np.load(original,allow_pickle=False) as data:
            centers=data['center_indices'].copy()
            positions=data['positions'].copy()
        provenance={'source_sha256':sha(image),'metadata_sha256':sha(metadata),'original_tokens_sha256':sha(original)}
        dest=args.output/'cases'/f'{case}.npz'
        marker=dest.with_suffix('.json')
        (_, _),(y0,y1),(x0,x1)=meta['box_zyx']
        ratios.append((x1-x0)/(y1-y0))
        if marker.exists():
            if json.loads(marker.read_text())!=provenance or not dest.exists():
                raise ValueError('Cached extraction provenance mismatch')
        else:
            volume=sitk.GetArrayFromImage(sitk.ReadImage(str(image))).astype(np.float32)
            if list(volume.shape)!=meta['shape_zyx'] or not np.isfinite(volume).all():
                raise ValueError('CT audit failed')
            images=[]
            for center in centers:
                channels=np.stack([volume[np.clip(center+d,0,len(volume)-1),y0:y1,x0:x1] for d in (-1,0,1)])
                channels=torch.from_numpy((np.clip(channels,-150.,250.)+150.)/400.)
                images.append(F.interpolate(square_pad(channels)[None],(224,224),mode='bilinear',align_corners=False)[0])
            normalized=(torch.stack(images)-encoder.imagenet_mean)/encoder.imagenet_std
            with torch.inference_mode():
                features=torch.cat([backbone(chunk.to(args.device)).flatten(1).cpu() for chunk in normalized.split(16)])
            features=F.normalize(features,dim=1).numpy().astype(np.float16)
            np.savez_compressed(dest,features=features,positions=positions,center_indices=centers)
            marker.write_text(json.dumps(provenance,indent=2)+'\n')
        with np.load(dest,allow_pickle=False) as data:
            assert data['features'].shape==(len(centers),512) and np.isfinite(data['features']).all()
            np.testing.assert_array_equal(data['center_indices'],centers)
            np.testing.assert_array_equal(data['positions'],positions)
        manifest.append({'case_id':case,'SeriesInstanceUID':meta['SeriesInstanceUID'],'sequence_path':str(dest.resolve())})
        print(f'letterbox {number+1}/{len(files)} verified',flush=True)
    pd.DataFrame(manifest).to_csv(args.output/'manifest.csv',index=False)
    audit={'cases':len(files),'paired_centers_verified':True,'source_hashes_verified':True,
           'crop_width_height_ratio_min_median_max':np.quantile(ratios,[0,.5,1]).tolist(),
           'preserves_pixel_aspect_ratio':True,'no_physical_resampling':True,'outcome_independent':True}
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    main()
