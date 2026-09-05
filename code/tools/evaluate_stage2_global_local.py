"""S3: train-only convex fusion of full-field and renal letterbox risks."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from build_renal_2p5d_program_cache import sha
from evaluate_resnet_sequence_models import load_sequences
from evaluate_stunet_trimodal_pooling_nested_cv import fit_outer_modality,select_bimodal_weights
from evaluate_renal_2p5d_program import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-cache',type=Path,default=Path('data/embeddings/vision/renal_2p5d_program_v1/full'))
    p.add_argument('--local-cache',type=Path,default=Path('data/embeddings/vision/renal_2p5d_letterbox_v1'))
    p.add_argument('--baseline',type=Path,default=Path('results_vision/fullfield_2p5d_adaptation_v1'))
    p.add_argument('--output',type=Path,default=Path('results_vision/stage2_global_local_v1'))
    args=p.parse_args()
    cohort=pd.read_csv(args.baseline/'cohort_common.csv').set_index('case_id')
    ids=cohort.index.tolist()
    assert len(ids)==75 and int(cohort.event.sum())==20 and ids==sorted(set(ids))
    sequences={'full':load_sequences(args.full_cache),'local':load_sequences(args.local_cache)}
    assert all(set(s)==set(ids) for s in sequences.values())
    arrays={name:np.stack([s[case][0].mean(axis=0) for case in ids]) for name,s in sequences.items()}
    assert all(x.shape==(75,512) and np.isfinite(x).all() for x in arrays.values())
    times,events=cohort.survival_days.to_numpy(float),cohort.event.to_numpy(int)
    contract={'script_sha256':sha(__file__),'n_cases':75,'events':20,'outer':'5x3 seed4049','inner':3,
              'full_contract_sha256':sha(args.full_cache.parent/'contract.json'),
              'local_contract_sha256':sha(args.local_cache/'contract.json'),
              'cohort_sha256':sha(args.baseline/'cohort_common.csv'),
              'pca':[4,8],'penalties':[.1,1.,10.,100.],'weight_step':.1,
              'files':{name:{case:sha(folder/'cases'/f'{case}.npz') for case in ids}
                       for name,folder in [('full',args.full_cache),('local',args.local_cache)]}}
    args.output.mkdir(parents=True,exist_ok=True)
    folds=args.output/'folds'; folds.mkdir(exist_ok=True)
    path=folds/'contract.json'
    if path.exists() and json.loads(path.read_text())!=contract: raise ValueError('Run contract changed')
    path.write_text(json.dumps(contract,indent=2)+'\n')
    public={k:v for k,v in contract.items() if k!='files'}; public['local_contract_sha256_complete']=sha(path)
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
            oof,risks,metadata={},{},{}
            for name,values in arrays.items():
                oof[name],risk,metadata[name]=fit_outer_modality(values,times,events,train,test,'vision',seed,[4,8],[.1,1.,10.,100.],3)
                risks[name]=risk.tolist()
            weights,score=select_bimodal_weights(np.column_stack([oof['full'],oof['local']]),times[train],events[train],.1)
            risks['global_local']=(np.column_stack([risks['full'],risks['local']])@weights).tolist()
            metadata['global_local']={'weight_full':float(weights[0]),'weight_local':float(weights[1]),'inner_cindex':score}
            saved={'ids':[ids[i] for i in test],'risks':risks,'fit':metadata}
            partial=path.with_suffix('.partial'); partial.write_text(json.dumps(saved)); partial.replace(path)
        for name,risk in risks.items():
            assert np.isfinite(risk).all()
            rows.extend({'model':name,'repeat':repeat,'fold':fold,'case_id':ids[i],'survival_days':float(times[i]),'event':int(events[i]),'risk':float(r)} for i,r in zip(test,risk))
            fits.append({'model':name,'repeat':repeat,'fold':fold,**metadata[name]})
        splits.extend({'case_id':ids[i],'repeat':repeat,'fold':fold,'partition':'train' if i in train else 'heldout'} for i in range(len(ids)))
        print(f'global-local {number+1}/15 complete',flush=True)
    pred=pd.DataFrame(rows)
    pred.to_csv(args.output/'heldout_predictions.csv',index=False)
    pd.DataFrame(fits).to_csv(args.output/'fitting_metrics.csv',index=False)
    pd.DataFrame(splits).to_csv(args.output/'splits.csv',index=False)
    print(json.dumps(summarize(pred,args.output,[('global_local','full')]),indent=2))


if __name__=='__main__': main()
