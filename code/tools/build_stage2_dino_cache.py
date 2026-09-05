"""S4 paired frozen DINOv2-S/14 and ResNet18 on identical stored 2.5D images."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18,ResNet18_Weights
from build_renal_2p5d_program_cache import sha

DINO_REVISION='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
DINO_URL='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-code',type=Path,default=Path('data/models/dinov2_source'))
    p.add_argument('--images',type=Path,default=Path('data/embeddings/vision/fullfield_2p5d_adaptation_v1'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--pilot',action='store_true')
    args=p.parse_args()
    revision=subprocess.check_output(['git','-C',str(args.source_code),'rev-parse','HEAD'],text=True).strip()
    if revision!=DINO_REVISION or subprocess.run(['git','-C',str(args.source_code),'diff','--quiet','HEAD']).returncode:
        raise ValueError('Expected pinned unmodified official DINOv2 source')
    sys.path.insert(0,str(args.source_code.resolve()))
    from dinov2.hub.backbones import dinov2_vits14
    torch.set_num_threads(1)
    torch.manual_seed(47)
    torch.hub.set_dir('data/models/torch')
    device=torch.device(args.device)
    dino=dinov2_vits14(pretrained=False)
    weights=torch.hub.load_state_dict_from_url(DINO_URL,map_location='cpu',weights_only=True)
    dino.load_state_dict(weights,strict=True)
    del weights
    resnet=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    models={'dinov2':dino.to(device).eval(),'resnet18':torch.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()}
    for model in models.values():
        for parameter in model.parameters(): parameter.requires_grad_(False)
    contract={'script_sha256':sha(__file__),'source_revision':revision,'dino_weights_url':DINO_URL,
              'dino_weights_sha256':sha('data/models/torch/checkpoints/dinov2_vits14_pretrain.pth'),
              'resnet_weights_sha256':sha('data/models/torch/checkpoints/resnet18-f37072fd.pth'),
              'images_contract_sha256':sha(args.images/'contract.json'),'n_tokens':16,'size':224,
              'dimensions':{'dinov2':384,'resnet18':512},'normalization':'ImageNet mean/std then per-token L2',
              'outcome_independent':True,'pilot':args.pilot,'license':'Apache-2.0 DINOv2 code and model card'}
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'contract.json'
    if path.exists() and json.loads(path.read_text())!=contract: raise ValueError('Extraction contract changed')
    path.write_text(json.dumps(contract,indent=2)+'\n')
    files=sorted((args.images/'images').glob('*.npz'))
    if len(files)!=75: raise ValueError('Expected all 75 audited image files')
    if args.pilot: files=files[:1]
    for name in models: (args.output/name/'cases').mkdir(parents=True,exist_ok=True)
    mean=torch.tensor([.485,.456,.406]).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
    started=time.monotonic()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    for number,source in enumerate(files):
        with np.load(source,allow_pickle=False) as data:
            images=torch.from_numpy(data['images'].astype(np.float32))
            centers=data['center_indices'].copy()
        assert images.shape==(16,3,224,224) and torch.isfinite(images).all()
        source_hash=sha(source)
        normalized=(images-mean)/std
        for name,model in models.items():
            dest=args.output/name/'cases'/source.name
            if dest.exists():
                with np.load(dest,allow_pickle=False) as data:
                    if str(data['source_sha256'])!=source_hash: raise ValueError('Cached image source changed')
                    features=data['features']
                    np.testing.assert_array_equal(data['center_indices'],centers)
            else:
                with torch.inference_mode():
                    features=torch.cat([model(chunk.to(device)).flatten(1).cpu() for chunk in normalized.split(4)])
                features=torch.nn.functional.normalize(features,dim=1).numpy().astype(np.float16)
                np.savez_compressed(dest,features=features,center_indices=centers,source_sha256=source_hash)
            assert features.shape==(16,contract['dimensions'][name]) and np.isfinite(features).all()
        print(f'paired DINO/ResNet {number+1}/{len(files)} verified',flush=True)
    audit={'cases':len(files),'n_tokens':16,'paired_source_images':True,'finite_features':True,
           'elapsed_seconds':time.monotonic()-started,'peak_cuda_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
           'pilot_only':args.pilot,'outcome_independent':True,'script_sha256':sha(__file__)}
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit),flush=True)


if __name__=='__main__': main()
