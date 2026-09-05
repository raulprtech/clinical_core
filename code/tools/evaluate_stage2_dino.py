"""S4 paired nested Cox comparison of frozen DINOv2 and ResNet18."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from build_renal_2p5d_program_cache import sha
from evaluate_renal_2p5d_program import cox_nested,summarize


def load_arrays(cache,ids):
    vectors={name:[] for name in ('resnet18','dinov2')}
    for case in ids:
        reference_centers,reference_hash=None,None
        for name,dimension in [('resnet18',512),('dinov2',384)]:
            with np.load(cache/name/'cases'/f'{case}.npz',allow_pickle=False) as data:
                features=data['features'].astype(np.float32)
                centers=data['center_indices']
                source_hash=str(data['source_sha256'])
            if features.shape!=(16,dimension) or not np.isfinite(features).all():
                raise ValueError('Invalid paired feature shape or values')
            if reference_centers is None: reference_centers,reference_hash=centers,source_hash
            else:
                np.testing.assert_array_equal(centers,reference_centers)
                if source_hash!=reference_hash: raise ValueError('Encoders used different input images')
            norms=np.linalg.norm(features,axis=1)
            if (norms<=0).any(): raise ValueError('Zero feature token')
            vectors[name].append((features/norms[:,None]).mean(axis=0))
    return {name:np.stack(rows) for name,rows in vectors.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('data/embeddings/vision/stage2_dino_v1'))
    p.add_argument('--baseline',type=Path,default=Path('results_vision/fullfield_2p5d_adaptation_v1'))
    p.add_argument('--output',type=Path,default=Path('results_vision/stage2_dino_cox_v1'))
    args=p.parse_args()
    extract=json.loads((args.cache/'contract.json').read_text())
    if extract['pilot']: raise ValueError('Cannot evaluate a pilot cache')
    cohort=pd.read_csv(args.baseline/'cohort_common.csv').set_index('case_id')
    ids=cohort.index.tolist()
    assert len(ids)==75 and ids==sorted(set(ids)) and int(cohort.event.sum())==20
    for name in ('resnet18','dinov2'):
        assert {p.stem for p in (args.cache/name/'cases').glob('*.npz')}==set(ids)
    arrays=load_arrays(args.cache,ids)
    times,events=cohort.survival_days.to_numpy(float),cohort.event.to_numpy(int)
    contract={'script_sha256':sha(__file__),'cache_contract_sha256':sha(args.cache/'contract.json'),
              'cohort_sha256':sha(args.baseline/'cohort_common.csv'),'n_cases':75,'events':20,
              'outer':'5x3 seed4049','inner':3,'pca':[4,8],'cox_alpha':[100.,10.,1.],
              'features_sha256':{name:{case:sha(args.cache/name/'cases'/f'{case}.npz') for case in ids} for name in arrays}}
    args.output.mkdir(parents=True,exist_ok=True)
    folds=args.output/'folds'; folds.mkdir(exist_ok=True)
    path=folds/'contract.json'
    if path.exists() and json.loads(path.read_text())!=contract: raise ValueError('Run contract changed')
    path.write_text(json.dumps(contract,indent=2)+'\n')
    public={k:v for k,v in contract.items() if k!='features_sha256'}; public['local_contract_sha256']=sha(path)
    (args.output/'provenance.json').write_text(json.dumps(public,indent=2)+'\n')
    cohort.reset_index().to_csv(args.output/'cohort_common.csv',index=False)
    rows,fits,splits=[],[],[]
    for number,(train,test) in enumerate(RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=4049).split(ids,events)):
        repeat,fold=divmod(number,5); seed=4049+100*repeat+fold
        path=folds/f'{repeat}_{fold}.json'
        if path.exists():
            saved=json.loads(path.read_text()); assert saved['ids']==[ids[i] for i in test]
            risks,metadata=saved['risks'],saved['fit']
        else:
            risks,metadata={},{}
            for name,values in arrays.items():
                risk,metadata[name]=cox_nested(values,times,events,train,test,seed)
                risks[name]=risk.tolist()
            saved={'ids':[ids[i] for i in test],'risks':risks,'fit':metadata}
            partial=path.with_suffix('.partial'); partial.write_text(json.dumps(saved)); partial.replace(path)
        for name,risk in risks.items():
            assert np.isfinite(risk).all()
            rows.extend({'model':name,'repeat':repeat,'fold':fold,'case_id':ids[i],'survival_days':float(times[i]),'event':int(events[i]),'risk':float(r)} for i,r in zip(test,risk))
            fits.append({'model':name,'repeat':repeat,'fold':fold,**metadata[name]})
        splits.extend({'case_id':ids[i],'repeat':repeat,'fold':fold,'partition':'train' if i in train else 'heldout'} for i in range(len(ids)))
        print(f'DINO/ResNet Cox {number+1}/15 complete',flush=True)
    pred=pd.DataFrame(rows)
    pred.to_csv(args.output/'heldout_predictions.csv',index=False)
    pd.DataFrame(fits).to_csv(args.output/'fitting_metrics.csv',index=False)
    pd.DataFrame(splits).to_csv(args.output/'splits.csv',index=False)
    print(json.dumps(summarize(pred,args.output,[('dinov2','resnet18')]),indent=2))


if __name__=='__main__': main()
