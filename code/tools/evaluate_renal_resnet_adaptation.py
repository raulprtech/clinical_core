"""Matched frozen/last-block ResNet18 experiment on renal 2.5D crops.

Cache only the immutable pretrained prefix (through layer3). Layer4 and the
linear survival head are initialized afresh within every training split.
Two-pass differentiation computes exact full-cohort Cox gradients with one
patient activation graph at a time. BatchNorm always uses pretrained statistics.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from build_renal_2p5d_program_cache import sha
from evaluate_resnet_sequence_models import load_targets, safe_cindex, seed_everything
from evaluate_renal_2p5d_program import summarize
def cox_ph_loss(risk, times, events):
    """Breslow partial likelihood with complete risk sets for tied times."""
    if float(events.sum()) <= 0:
        raise ValueError('Cox training requires events')
    at_risk = times[None, :] >= times[:, None]
    denominator = torch.logsumexp(risk[None, :].expand(len(risk), -1).masked_fill(~at_risk, -torch.inf), dim=1)
    return ((denominator - risk) * events).sum() / events.sum()


def two_pass_backward(model, xs, times, events, device):
    """Accumulate exact Cox parameter gradients without changing model state."""
    model.eval()  # Deterministic, identical forwards, frozen BatchNorm.
    with torch.no_grad():
        risk = torch.stack([model(x.to(device)) for x in xs])
    risk.requires_grad_(True)
    loss = cox_ph_loss(risk, torch.as_tensor(times, dtype=torch.float32, device=device),
                       torch.as_tensor(events, dtype=torch.float32, device=device))
    gradients, = torch.autograd.grad(loss, risk)
    for x, coefficient in zip(xs, gradients):
        model(x.to(device)).backward(coefficient)
    return float(loss.detach())


class LastBlockSurvival(nn.Module):
    def __init__(self, layer4, tune):
        super().__init__()
        self.layer4 = copy.deepcopy(layer4)
        for p in self.layer4.parameters():
            p.requires_grad_(tune)
        # Keep affine BN parameters and running statistics frozen in both arms.
        for m in self.layer4.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad_(False)
        self.head = nn.Linear(512, 1, bias=False)

    def forward(self, x):
        features = self.layer4(x).mean(dim=(0, 2, 3))
        features = torch.nn.functional.normalize(features, dim=0)
        return self.head(features).squeeze()


def new_model(layer4, tune, seed, device):
    seed_everything(seed)
    model = LastBlockSurvival(layer4, tune).to(device).eval()
    groups = [{'params': list(model.head.parameters()), 'lr': 1e-3}]
    if tune:
        groups.append({'params': [p for p in model.layer4.parameters() if p.requires_grad], 'lr': 1e-5})
    return model, torch.optim.AdamW(groups, weight_decay=1e-3)


def train_epoch(model, optimizer, xs, times, events, device):
    optimizer.zero_grad(set_to_none=True)
    loss = two_pass_backward(model, xs, times, events, device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    optimizer.step()
    return loss


def predict(model, xs, device):
    model.eval()
    with torch.no_grad():
        return np.array([float(model(x.to(device)).cpu()) for x in xs])


def prepare(cache, prefix_cache, pretrained, device):
    prefix_cache.mkdir(parents=True, exist_ok=True)
    source_contract = sha(cache / 'contract.json')
    prefix = nn.Sequential(*list(pretrained.children())[:7]).to(device).eval()
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    image_files = sorted((cache / 'images').glob('*.npz'))
    for i, source in enumerate(image_files):
        dest = prefix_cache / source.name
        source_hash = sha(source)
        if dest.exists():
            with np.load(dest, allow_pickle=False) as d:
                if str(d['source_sha256']) == source_hash and str(d['contract_sha256']) == source_contract:
                    continue
            raise ValueError('Prefix cache provenance mismatch: ' + source.stem)
        with np.load(source, allow_pickle=False) as d:
            images = torch.from_numpy(d['images'].astype(np.float32))
        with torch.inference_mode():
            feature = torch.cat([prefix(chunk.to(device)).cpu()
                                 for chunk in ((images - mean) / std).split(8)])
        np.savez_compressed(dest, features=feature.numpy().astype(np.float16),
                            source_sha256=source_hash, contract_sha256=source_contract)
        print(f'prefix {i+1}/{len(image_files)} cached', flush=True)
    del prefix
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--prefix-cache', type=Path, required=True)
    p.add_argument('--targets', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir('data/models/torch')
    device = torch.device(args.device)
    pretrained = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    prepare(args.cache, args.prefix_cache, pretrained, device)
    if args.prepare_only:
        return
    if args.targets is None or args.output is None:
        raise ValueError('Training requires targets and output')
    args.output.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.targets)
    ids = sorted(set(targets.index) & {p.stem for p in args.prefix_cache.glob('*.npz')})
    cohort = targets.loc[ids]
    cohort = cohort[(cohort.survival_days > 0) & cohort.event.isin([0, 1])]
    ids = cohort.index.tolist()
    times, events = cohort.survival_days.to_numpy(float), cohort.event.to_numpy(int)
    xs = []
    for case in ids:
        with np.load(args.prefix_cache / f'{case}.npz', allow_pickle=False) as d:
            xs.append(torch.from_numpy(d['features'].astype(np.float32)))
    layer4 = pretrained.layer4.cpu()
    # Avoid touching pretrained layers retained by an earlier prefix reference.
    layer4 = copy.deepcopy(layer4)
    del pretrained
    if args.pilot:
        train, heldout = next(StratifiedKFold(5, shuffle=True, random_state=4049).split(ids, events))
        model, optimizer = new_model(layer4, True, 4049, device)
        before = next(p for p in model.layer4.parameters() if p.requires_grad).detach().clone()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        loss = train_epoch(model, optimizer, [xs[i] for i in train], times[train], events[train], device)
        after = next(p for p in model.layer4.parameters() if p.requires_grad).detach()
        output = {'n_cases': len(ids), 'n_train': len(train), 'loss': loss,
                  'epoch_seconds': time.monotonic()-started,
                  'layer4_max_change': float((after-before).abs().max()),
                  'peak_cuda_mib': torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
                  'heldout_finite': bool(np.isfinite(predict(model, [xs[i] for i in heldout], device)).all()),
                  'pilot_only': True, 'no_performance_selection': True}
        (args.output / 'technical_pilot.json').write_text(json.dumps(output, indent=2)+'\n')
        print(json.dumps(output), flush=True)
        return
    contract = {'script_sha256': sha(__file__), 'targets_sha256': sha(args.targets),
                'cache_contract_sha256': sha(args.cache / 'contract.json'), 'n_cases': len(ids),
                'events': int(events.sum()), 'outer': '5x3', 'inner': 3, 'epochs': [1, 3, 5],
                'lr_head': .001, 'lr_layer4': .00001, 'seed': 4049,
                'frozen_batchnorm': True, 'n_tokens': 16,
                'prefixes_sha256': {case: sha(args.prefix_cache / f'{case}.npz') for case in ids}}
    # Patient-keyed contract stays local with per-fold checkpoints.
    folds_dir = args.output / 'folds'
    folds_dir.mkdir(exist_ok=True)
    path = folds_dir / 'contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Training contract changed; use a new output directory')
    path.write_text(json.dumps(contract, indent=2)+'\n')
    public_contract = {k:v for k,v in contract.items() if k != 'prefixes_sha256'}
    public_contract['local_contract_sha256'] = sha(path)
    (args.output / 'provenance.json').write_text(json.dumps(public_contract, indent=2)+'\n')
    cohort.reset_index().to_csv(args.output / 'cohort_common.csv', index=False)
    rows, fitting, splits = [], [], []
    for number, (train, heldout) in enumerate(RepeatedStratifiedKFold(
            n_splits=5, n_repeats=3, random_state=4049).split(ids, events)):
        repeat, fold = divmod(number, 5)
        seed = 4049 + 100*repeat + fold
        splits.extend({'case_id': ids[i], 'repeat': repeat, 'fold': fold,
                       'partition': 'train' if i in train else 'heldout'} for i in range(len(ids)))
        for name, tune in [('frozen', False), ('adapted', True)]:
            checkpoint = folds_dir / f'{repeat}_{fold}_{name}.json'
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                if saved['ids'] != [ids[i] for i in heldout]:
                    raise ValueError('Checkpoint heldout IDs differ')
            else:
                epoch_scores = {1: [], 3: [], 5: []}
                for inner, (itr, iva) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(train, events[train])):
                    tr, va = train[itr], train[iva]
                    model, optimizer = new_model(layer4, tune, seed*1000+inner, device)
                    for epoch in range(1, 6):
                        train_epoch(model, optimizer, [xs[i] for i in tr], times[tr], events[tr], device)
                        if epoch in epoch_scores:
                            risk = predict(model, [xs[i] for i in va], device)
                            epoch_scores[epoch].append(safe_cindex(times[va], risk, events[va]))
                    del model, optimizer
                selected = max(epoch_scores, key=lambda e: float(np.mean(epoch_scores[e])))
                model, optimizer = new_model(layer4, tune, seed*1000+29, device)
                for _ in range(selected):
                    train_epoch(model, optimizer, [xs[i] for i in train], times[train], events[train], device)
                risk = predict(model, [xs[i] for i in heldout], device)
                del model, optimizer
                saved = {'ids': [ids[i] for i in heldout], 'risk': risk.tolist(), 'epochs': selected,
                         'inner_scores': epoch_scores}
                partial = checkpoint.with_suffix('.partial')
                partial.write_text(json.dumps(saved))
                partial.replace(checkpoint)
            rows.extend({'model': name, 'repeat': repeat, 'fold': fold, 'case_id': ids[i],
                         'survival_days': float(times[i]), 'event': int(events[i]), 'risk': r}
                        for i,r in zip(heldout, saved['risk']))
            fitting.append({'model': name, 'repeat': repeat, 'fold': fold, 'epochs': saved['epochs']})
            print(f'adaptation {number+1}/15 {name} complete', flush=True)
    pred = pd.DataFrame(rows)
    pred.to_csv(args.output/'heldout_predictions.csv', index=False)
    pd.DataFrame(splits).to_csv(args.output/'splits.csv', index=False)
    pd.DataFrame(fitting).to_csv(args.output/'fitting_metrics.csv', index=False)
    print(json.dumps(summarize(pred, args.output, [('adapted', 'frozen')]), indent=2), flush=True)


if __name__ == '__main__':
    main()
