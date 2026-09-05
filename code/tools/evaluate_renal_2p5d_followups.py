"""Predeclared post-E1 follow-ups: token moments and incremental clinical fusion."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from build_renal_2p5d_program_cache import sha
from evaluate_resnet_sequence_models import load_sequences, load_targets
from evaluate_renal_2p5d_program import cox_nested, summarize
from evaluate_stunet_trimodal_pooling_nested_cv import fit_outer_modality, select_bimodal_weights
from evaluate_trimodal_fusion import load_indexed_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--features', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=['moments', 'fusion', 'fusion_moments'], required=True)
    args = p.parse_args()
    sets = {a: load_sequences(args.cache/a) for a in ['full', 'renal_crop']}
    targets = load_targets(args.targets)
    radio = pd.read_csv(args.cache/'radiomics_2d.csv').set_index('case_id')
    ids = set(targets.index) & set(radio.index) & set(sets['full']) & set(sets['renal_crop'])
    if args.kind.startswith('fusion'):
        if args.features is None:
            raise ValueError('Fusion requires features')
        tabular = load_indexed_csv(args.features, {'case_id'}).astype(float)
        ids &= set(tabular.index)
    cohort = targets.loc[sorted(ids)]
    cohort = cohort[(cohort.survival_days>0)&cohort.event.isin([0,1])]
    ids = cohort.index.tolist()
    times, events = cohort.survival_days.to_numpy(float), cohort.event.to_numpy(int)
    arrays = {a: np.stack([s[c][0].mean(axis=0) for c in ids]) for a,s in sets.items()}
    if args.kind == 'moments':
        arrays.update({a+'_moments': np.stack([np.r_[s[c][0].mean(axis=0), s[c][0].std(axis=0)]
                                             for c in ids]) for a,s in sets.items()})
        comparisons = [('full_moments','full'), ('renal_crop_moments','renal_crop')]
    else:
        arrays = {'tabular': tabular.loc[ids].to_numpy(float),
                  'full': arrays['full'], 'radiomics': radio.loc[ids].to_numpy(float)}
        comparisons = [('tabular_full','tabular'), ('tabular_radiomics','tabular')]
        if args.kind == 'fusion_moments':
            del arrays['radiomics']
            arrays['full_moments'] = np.stack([np.r_[sets['full'][c][0].mean(axis=0),
                                                     sets['full'][c][0].std(axis=0)] for c in ids])
            comparisons = [('tabular_full_moments','tabular_full'), ('tabular_full_moments','tabular')]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'folds').mkdir(exist_ok=True)
    contract = {'kind': args.kind, 'n_cases': len(ids), 'events': int(events.sum()),
                'script_sha256': sha(__file__), 'targets_sha256': sha(args.targets),
                'cache_contract_sha256': sha(args.cache/'contract.json'),
                'radiomics_sha256': sha(args.cache/'radiomics_2d.csv'),
                'features_sha256': sha(args.features) if args.features else None,
                'outer': '5x3, seed4049', 'inner': 3, 'hypotheses': comparisons,
                'arguments': {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    dest = args.output/'provenance.json'
    if dest.exists() and json.loads(dest.read_text()) != json.loads(json.dumps(contract)):
        raise ValueError('Existing run contract differs')
    dest.write_text(json.dumps(contract,indent=2)+'\n')
    cohort.reset_index().to_csv(args.output/'cohort_common.csv',index=False)
    rows, fits, splits = [], [], []
    for n, (train,test) in enumerate(RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=4049).split(ids,events)):
        repeat, fold = divmod(n,5)
        seed = 4049 + repeat*100 + fold
        checkpoint = args.output/'folds'/f'{repeat}_{fold}.json'
        if checkpoint.exists():
            done = json.loads(checkpoint.read_text())
            if done['ids'] != [ids[i] for i in test]:
                raise ValueError('Cached heldout patients differ')
            risks, meta = done['risks'], done['fit']
        else:
            risks, meta, oof = {}, {}, {}
            for name, x in arrays.items():
                if args.kind == 'moments':
                    risk, details = cox_nested(x,times,events,train,test,seed)
                else:
                    oof[name], risk, details = fit_outer_modality(
                        x,times,events,train,test,'tabular' if name=='tabular' else 'vision',
                        seed,[4,8],[.1,1.,10.,100.],3)
                risks[name], meta[name] = risk.tolist(), details
            if args.kind.startswith('fusion'):
                for other in [x for x in arrays if x != 'tabular']:
                    weights, inner_score = select_bimodal_weights(
                        np.column_stack([oof['tabular'],oof[other]]),times[train],events[train],.1)
                    name = 'tabular_'+other
                    risks[name] = (np.column_stack([risks['tabular'],risks[other]])@weights).tolist()
                    meta[name] = {'weight_tabular': float(weights[0]), 'weight_other': float(weights[1]),
                                  'inner_cindex': inner_score}
            done = {'ids': [ids[i] for i in test], 'risks': risks, 'fit':meta}
            partial = checkpoint.with_suffix('.partial')
            partial.write_text(json.dumps(done))
            partial.replace(checkpoint)
        for name,risk in risks.items():
            if not np.isfinite(risk).all():
                raise ValueError('Nonfinite prediction')
            rows.extend({'model':name,'repeat':repeat,'fold':fold,'case_id':ids[i],
                         'survival_days':float(times[i]),'event':int(events[i]),'risk':float(r)}
                        for i,r in zip(test,risk))
            fits.append({'model':name,'repeat':repeat,'fold':fold,**meta[name]})
        splits.extend({'case_id':ids[i],'repeat':repeat,'fold':fold,
                       'partition':'train' if i in train else 'heldout'} for i in range(len(ids)))
        print(f'{args.kind} {n+1}/15 complete',flush=True)
    pred = pd.DataFrame(rows)
    pred.to_csv(args.output/'heldout_predictions.csv',index=False)
    pd.DataFrame(splits).to_csv(args.output/'splits.csv',index=False)
    pd.DataFrame(fits).to_csv(args.output/'fitting_metrics.csv',index=False)
    print(json.dumps(summarize(pred,args.output,comparisons),indent=2),flush=True)


if __name__ == '__main__':
    main()
