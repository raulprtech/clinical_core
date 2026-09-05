"""F7: matched Mamba letterbox ablation using unchanged E2 controls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import RepeatedStratifiedKFold
from build_renal_2p5d_program_cache import sha
from evaluate_renal_2p5d_program import summarize
from evaluate_resnet_sequence_models import load_sequences, pad_sequences
from evaluate_resnet_sequence_nested_cv import select_epochs_inner, refit_and_predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    cli=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base=json.loads((cli.baseline/'provenance.json').read_text())
    extract=json.loads((cli.cache/'contract.json').read_text())
    if extract['parent_contract_sha256']!=base['cache_contract_sha256']:
        raise ValueError('Letterbox and controls have different source contracts')
    expected={'kind':'mamba','use_position':False,'inner_folds':3,'patience':20,'epochs':200,
              'lr':.001,'weight_decay':.001,'dropout':.1,'model_dim':128,'attention_dim':64,
              'state_dim':16,'mamba_blocks':2,'outer_repeats':3}
    for name,value in expected.items():
        if base['arguments'][name]!=value:
            raise ValueError('E2 control configuration differs: '+name)
    args=argparse.Namespace(**expected)
    cohort=pd.read_csv(cli.baseline/'cohort_common.csv').set_index('case_id')
    ids=cohort.index.tolist()
    assert len(ids)==75 and ids==sorted(set(ids)) and int(cohort.event.sum())==20
    seq=load_sequences(cli.cache)
    assert set(seq)==set(ids)
    times,events=cohort.survival_days.to_numpy(float),cohort.event.to_numpy(int)
    assert (times>0).all() and np.isfinite(times).all()
    f,pos,mask=pad_sequences(seq,ids)
    baseline=pd.read_csv(cli.baseline/'heldout_predictions.csv')
    controls=baseline[baseline.model.isin(['full','renal_crop'])].copy()
    assert len(controls)==450 and not controls.duplicated(['case_id','repeat','model']).any()
    split_reference=pd.read_csv(cli.baseline/'splits.csv')
    cli.output.mkdir(parents=True,exist_ok=True)
    folds=cli.output/'folds'
    folds.mkdir(exist_ok=True)
    contract={'script_sha256':sha(__file__),'baseline_provenance_sha256':sha(cli.baseline/'provenance.json'),
              'baseline_predictions_sha256':sha(cli.baseline/'heldout_predictions.csv'),
              'baseline_splits_sha256':sha(cli.baseline/'splits.csv'),
              'baseline_cohort_sha256':sha(cli.baseline/'cohort_common.csv'),
              'cache_contract_sha256':sha(cli.cache/'contract.json'),'n_cases':75,'events':20,
              'outer':'5x3, seed4049','parameters':expected,'controls_reused_not_retrained':['full','renal_crop'],
              'sequences_sha256':{case:sha(cli.cache/'cases'/f'{case}.npz') for case in ids}}
    path=folds/'contract.json'
    if path.exists() and json.loads(path.read_text())!=contract:
        raise ValueError('Run contract changed; use a new output directory')
    path.write_text(json.dumps(contract,indent=2)+'\n')
    public={k:v for k,v in contract.items() if k!='sequences_sha256'}
    public['local_contract_sha256']=sha(path)
    (cli.output/'provenance.json').write_text(json.dumps(public,indent=2)+'\n')
    cohort.reset_index().to_csv(cli.output/'cohort_common.csv',index=False)
    rows, fits=[],[]
    for number,(train,heldout) in enumerate(RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=4049).split(ids,events)):
        repeat,fold=divmod(number,5)
        seed=4049+100*repeat+fold
        ref=split_reference[(split_reference.repeat==repeat)&(split_reference.fold==fold)]
        assert set(ref.loc[ref.partition=='train','case_id'])=={ids[i] for i in train}
        assert set(ref.loc[ref.partition=='heldout','case_id'])=={ids[i] for i in heldout}
        path=folds/f'{repeat}_{fold}.json'
        if path.exists():
            saved=json.loads(path.read_text())
            assert saved['ids']==[ids[i] for i in heldout]
            risk,fit=np.array(saved['risk']),saved['fit']
        else:
            epoch,inner_epochs,scores=select_epochs_inner('mamba',f,pos,mask,times,events,train,seed,args,torch.device(cli.device))
            risk,parameters=refit_and_predict('mamba',f,pos,mask,times,events,train,heldout,epoch,
                                             seed*1000+29,args,torch.device(cli.device))
            assert np.isfinite(risk).all()
            fit={'epochs':epoch,'inner_epochs':inner_epochs,'inner_cindex':float(np.mean(scores)),'parameters':parameters}
            saved={'ids':[ids[i] for i in heldout],'risk':risk.tolist(),'fit':fit}
            partial=path.with_suffix('.partial')
            partial.write_text(json.dumps(saved))
            partial.replace(path)
        rows.extend({'model':'renal_letterbox','repeat':repeat,'fold':fold,'case_id':ids[i],
                     'survival_days':float(times[i]),'event':int(events[i]),'risk':float(r)} for i,r in zip(heldout,risk))
        fits.append({'model':'renal_letterbox','repeat':repeat,'fold':fold,**fit})
        print(f'letterbox Mamba {number+1}/15 complete',flush=True)
    pred=pd.concat([controls,pd.DataFrame(rows)],ignore_index=True)
    pred.to_csv(cli.output/'heldout_predictions.csv',index=False)
    split_reference.to_csv(cli.output/'splits.csv',index=False)
    old_fits=pd.read_csv(cli.baseline/'fitting_metrics.csv')
    old_fits=old_fits[old_fits.model.isin(['full','renal_crop'])].copy()
    old_fits['reused_control']=True
    new_fits=pd.DataFrame(fits)
    new_fits['reused_control']=False
    pd.concat([old_fits,new_fits],ignore_index=True).to_csv(cli.output/'fitting_metrics.csv',index=False)
    print(json.dumps(summarize(pred,cli.output,[('renal_letterbox','renal_crop'),('renal_letterbox','full')]),indent=2),flush=True)


if __name__=='__main__':
    main()
