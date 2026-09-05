"""Internal cohort expansion: fixed ResNet token mean vs mean+std, no masks."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from evaluate_resnet_sequence_models import load_sequences, load_targets
from evaluate_renal_2p5d_program import cox_nested, summarize
from build_renal_2p5d_program_cache import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence-dir',type=Path,required=True)
    p.add_argument('--targets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    seq=load_sequences(args.sequence_dir)
    targets=load_targets(args.targets)
    cohort=targets.loc[sorted(set(targets.index)&set(seq))]
    cohort=cohort[(cohort.survival_days>0)&cohort.event.isin([0,1])]
    ids=cohort.index.tolist()
    times,events=cohort.survival_days.to_numpy(float),cohort.event.to_numpy(int)
    mean=np.stack([seq[c][0].mean(axis=0) for c in ids])
    std=np.stack([seq[c][0].std(axis=0) for c in ids])
    arrays={'mean':mean,'moments':np.concatenate([mean,std],axis=1)}
    args.output.mkdir(parents=True,exist_ok=True)
    folds=args.output/'folds'
    folds.mkdir(exist_ok=True)
    contract={'script_sha256':sha(__file__),'sequence_manifest_sha256':sha(args.sequence_dir/'manifest.csv'),
              'targets_sha256':sha(args.targets),'n_cases':len(ids),'events':int(events.sum()),
              'outer':'5x3 seed4049','inner':3,'components':[4,8],'alphas':[100.,10.,1.],
              'arguments':{k:str(v) for k,v in vars(args).items()}}
    dest=args.output/'provenance.json'
    if dest.exists() and json.loads(dest.read_text())!=contract:
        raise ValueError('Run contract differs')
    dest.write_text(json.dumps(contract,indent=2)+'\n')
    cohort.reset_index().to_csv(args.output/'cohort_common.csv',index=False)
    rows,fits,splits=[],[],[]
    for n,(train,test) in enumerate(RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=4049).split(ids,events)):
        repeat,fold=divmod(n,5)
        for name,x in arrays.items():
            checkpoint=folds/f'{repeat}_{fold}_{name}.json'
            if checkpoint.exists():
                saved=json.loads(checkpoint.read_text())
                if saved['ids']!=[ids[i] for i in test]:
                    raise ValueError('Cached IDs differ')
                risk,meta=saved['risk'],saved['fit']
            else:
                risk,meta=cox_nested(x,times,events,train,test,4049+100*repeat+fold)
                saved={'ids':[ids[i] for i in test],'risk':risk.tolist(),'fit':meta}
                tmp=checkpoint.with_suffix('.partial')
                tmp.write_text(json.dumps(saved))
                tmp.replace(checkpoint)
            rows.extend({'model':name,'repeat':repeat,'fold':fold,'case_id':ids[i],
                         'survival_days':float(times[i]),'event':int(events[i]),'risk':float(r)} for i,r in zip(test,risk))
            fits.append({'model':name,'repeat':repeat,'fold':fold,**meta})
        splits.extend({'case_id':ids[i],'repeat':repeat,'fold':fold,'partition':'train' if i in train else 'heldout'} for i in range(len(ids)))
        print(f'fullfield214 {n+1}/15 complete',flush=True)
    pred=pd.DataFrame(rows)
    pred.to_csv(args.output/'heldout_predictions.csv',index=False)
    pd.DataFrame(fits).to_csv(args.output/'fitting_metrics.csv',index=False)
    pd.DataFrame(splits).to_csv(args.output/'splits.csv',index=False)
    print(json.dumps(summarize(pred,args.output,[('moments','mean')]),indent=2),flush=True)


if __name__=='__main__':
    main()
