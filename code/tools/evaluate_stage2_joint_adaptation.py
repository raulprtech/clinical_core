"""S1/S2: nested early-stopped linear or joint Mamba/ResNet adaptation."""
from __future__ import annotations
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from build_renal_2p5d_program_cache import sha
from evaluate_renal_resnet_adaptation import LastBlockSurvival, train_epoch as linear_train_epoch, predict as linear_predict, cox_ph_loss
from evaluate_resnet_sequence_models import seed_everything, safe_cindex
from evaluate_resnet_sequence_nested_cv import make_model
from evaluate_renal_2p5d_program import summarize


class JointMamba(LastBlockSurvival):
    def __init__(self,layer4,tune):
        super().__init__(layer4,tune)
        config=argparse.Namespace(model_dim=128,attention_dim=64,state_dim=16,
                                  mamba_blocks=2,dropout=0.,use_position=False)
        self.head=make_model('mamba',config,torch.device('cpu'))
        self._tokens_cache={}

    def encode(self,x):
        return torch.nn.functional.normalize(self.layer4(x).mean(dim=(2,3)),dim=1)

    def risk_from_tokens(self,tokens):
        positions=tokens.new_zeros(tokens.shape[:2])
        mask=torch.ones_like(positions,dtype=torch.bool)
        return self.head(tokens,positions,mask)

    def forward(self,x):
        return self.risk_from_tokens(self.encode(x)[None]).squeeze()


def detached_tokens(model,xs,device):
    tune=any(p.requires_grad for p in model.layer4.parameters())
    rows=[]
    with torch.no_grad():
        for x in xs:
            if not tune and id(x) in model._tokens_cache:
                tokens=model._tokens_cache[id(x)]
            else:
                tokens=model.encode(x.to(device))
                if not tune: model._tokens_cache[id(x)]=tokens
            rows.append(tokens)
    return torch.stack(rows),tune


def batched_mamba_backward(model,xs,times,events,device):
    """Exact chain rule: batched head gradient then per-patient encoder VJP."""
    model.eval()
    tokens,tune=detached_tokens(model,xs,device)
    tokens.requires_grad_(tune)
    risk=model.risk_from_tokens(tokens)
    loss=cox_ph_loss(risk,torch.as_tensor(times,dtype=torch.float32,device=device),
                     torch.as_tensor(events,dtype=torch.float32,device=device))
    loss.backward()
    if tune:
        for x,gradient in zip(xs,tokens.grad):
            model.encode(x.to(device)).backward(gradient)
    return float(loss.detach())


def train_epoch(model,optimizer,xs,times,events,device):
    if not isinstance(model,JointMamba):
        return linear_train_epoch(model,optimizer,xs,times,events,device)
    optimizer.zero_grad(set_to_none=True)
    loss=batched_mamba_backward(model,xs,times,events,device)
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
    optimizer.step()
    return loss


def predict(model,xs,device):
    if not isinstance(model,JointMamba): return linear_predict(model,xs,device)
    model.eval()
    with torch.no_grad():
        tokens,_=detached_tokens(model,xs,device)
        return model.risk_from_tokens(tokens).cpu().numpy()


def initialize(layer4,tune,head,seed,device):
    seed_everything(seed)
    model=(JointMamba if head=='mamba' else LastBlockSurvival)(layer4,tune).to(device).eval()
    groups=[{'params':list(model.head.parameters()),'lr':.001}]
    if tune:
        groups.append({'params':[p for p in model.layer4.parameters() if p.requires_grad],'lr':1e-5})
    return model,torch.optim.AdamW(groups,weight_decay=.001)


def select_epoch(layer4,tune,head,xs,times,events,train,val,seed,device,max_epochs,min_epochs,patience):
    model,optimizer=initialize(layer4,tune,head,seed,device)
    best_epoch,best_score,stale=1,-float('inf'),0
    curve=[]
    for epoch in range(1,max_epochs+1):
        loss=train_epoch(model,optimizer,[xs[i] for i in train],times[train],events[train],device)
        risk=predict(model,[xs[i] for i in val],device)
        score=safe_cindex(times[val],risk,events[val])
        if not np.isfinite([loss,score]).all():
            raise ValueError('Nonfinite inner loss or C-index')
        curve.append({'epoch':epoch,'train_loss':loss,'validation_cindex':float(score)})
        if score>best_score+1e-8:
            best_epoch,best_score,stale=epoch,float(score),0
        else:
            stale+=1
        if epoch>=min_epochs and stale>=patience:
            break
    return {'best_epoch':best_epoch,'best_cindex':best_score,'epochs_run':epoch,
            'stop_reason':'patience' if epoch>=min_epochs and stale>=patience else 'max_epochs',
            'curve':curve}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('results_vision/fullfield_2p5d_adaptation_v1'))
    p.add_argument('--prefix-cache',type=Path,default=Path('data/embeddings/vision/fullfield_2p5d_prefix_v1'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--head',choices=['linear','mamba'],required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--max-epochs',type=int,default=100)
    p.add_argument('--min-epochs',type=int,default=20)
    p.add_argument('--patience',type=int,default=15)
    p.add_argument('--pilot',action='store_true')
    args=p.parse_args()
    if not (1<=args.min_epochs<=args.max_epochs and args.patience>=1):
        raise ValueError('Invalid stopping limits')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir('data/models/torch')
    device=torch.device(args.device)
    cohort=pd.read_csv(args.baseline/'cohort_common.csv').set_index('case_id')
    ids=cohort.index.tolist()
    assert len(ids)==75 and ids==sorted(set(ids)) and int(cohort.event.sum())==20
    source=json.loads((args.baseline/'folds'/'contract.json').read_text())
    xs=[]
    for case in ids:
        path=args.prefix_cache/f'{case}.npz'
        if sha(path)!=source['prefixes_sha256'][case]:
            raise ValueError('Audited prefix changed')
        with np.load(path,allow_pickle=False) as data:
            x=data['features'].astype(np.float32)
            assert x.shape==(16,256,14,14) and np.isfinite(x).all()
            xs.append(torch.from_numpy(x))
    times,events=cohort.survival_days.to_numpy(float),cohort.event.to_numpy(int)
    pretrained=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    layer4=copy.deepcopy(pretrained.layer4).eval()
    del pretrained
    args.output.mkdir(parents=True,exist_ok=True)
    if args.pilot:
        train,test=next(StratifiedKFold(5,shuffle=True,random_state=4049).split(ids,events))
        model,optimizer=initialize(layer4,True,args.head,4049,device)
        parameter=next(p for p in model.layer4.parameters() if p.requires_grad)
        before=parameter.detach().clone()
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        loss=train_epoch(model,optimizer,[xs[i] for i in train],times[train],events[train],device)
        pilot={'head':args.head,'script_sha256':sha(__file__),'n_train':len(train),'loss':loss,'epoch_seconds':time.monotonic()-started,
               'layer4_max_change':float((parameter.detach()-before).abs().max()),
               'heldout_finite':bool(np.isfinite(predict(model,[xs[i] for i in test],device)).all()),
               'peak_cuda_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
               'technical_only':True,'no_performance_selection':True}
        (args.output/'technical_pilot.json').write_text(json.dumps(pilot,indent=2)+'\n')
        print(json.dumps(pilot),flush=True)
        return
    contract={'script_sha256':sha(__file__),'source_contract_sha256':sha(args.baseline/'folds'/'contract.json'),
              'head':args.head,'n_cases':75,'events':20,'outer':'5x3 seed4049','inner':3,
              'max_epochs':args.max_epochs,'min_epochs':args.min_epochs,'patience':args.patience,
              'selection':'rounded median of best inner epochs','dropout':0,'frozen_batchnorm':True,
              'lr_head':.001,'lr_layer4':1e-5,'weight_decay':.001,'n_tokens':16}
    path=args.output/'provenance.json'
    if path.exists() and json.loads(path.read_text())!=contract:
        raise ValueError('Run contract changed; choose a new output directory')
    path.write_text(json.dumps(contract,indent=2)+'\n')
    folds=args.output/'folds'; folds.mkdir(exist_ok=True)
    cohort.reset_index().to_csv(args.output/'cohort_common.csv',index=False)
    rows,fits,splits,diagnostics=[],[],[],[]
    for number,(train,test) in enumerate(RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=4049).split(ids,events)):
        repeat,fold=divmod(number,5); seed=4049+100*repeat+fold
        splits.extend({'case_id':ids[i],'repeat':repeat,'fold':fold,'partition':'train' if i in train else 'heldout'} for i in range(len(ids)))
        for name,tune in [('frozen',False),('adapted',True)]:
            inner_results=[]
            for inner,(itr,iva) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(train,events[train])):
                path=folds/f'{repeat}_{fold}_{name}_inner{inner}.json'
                if path.exists(): result=json.loads(path.read_text())
                else:
                    result=select_epoch(layer4,tune,args.head,xs,times,events,train[itr],train[iva],
                                        seed*1000+inner,device,args.max_epochs,args.min_epochs,args.patience)
                    partial=path.with_suffix('.partial'); partial.write_text(json.dumps(result)); partial.replace(path)
                inner_results.append(result)
                diagnostics.extend({'model':name,'repeat':repeat,'fold':fold,'inner':inner,**row} for row in result['curve'])
                print(f'{args.head} {number+1}/15 {name} inner{inner} best={result["best_epoch"]} stop={result["epochs_run"]}',flush=True)
            selected=max(1,int(np.rint(np.median([r['best_epoch'] for r in inner_results]))))
            path=folds/f'{repeat}_{fold}_{name}_outer.json'
            if path.exists():
                saved=json.loads(path.read_text()); assert saved['ids']==[ids[i] for i in test] and saved['epoch']==selected
                risk=saved['risk']
            else:
                model,optimizer=initialize(layer4,tune,args.head,seed*1000+29,device)
                for _ in range(selected): train_epoch(model,optimizer,[xs[i] for i in train],times[train],events[train],device)
                risk=predict(model,[xs[i] for i in test],device).tolist()
                assert np.isfinite(risk).all()
                saved={'ids':[ids[i] for i in test],'epoch':selected,'risk':risk}
                partial=path.with_suffix('.partial'); partial.write_text(json.dumps(saved)); partial.replace(path)
                del model,optimizer
            rows.extend({'model':name,'repeat':repeat,'fold':fold,'case_id':ids[i],'survival_days':float(times[i]),'event':int(events[i]),'risk':float(r)} for i,r in zip(test,risk))
            fits.append({'model':name,'repeat':repeat,'fold':fold,'epochs':selected,
                         'inner_best_epochs':json.dumps([r['best_epoch'] for r in inner_results]),
                         'inner_stopped_at_cap':sum(r['stop_reason']=='max_epochs' for r in inner_results)})
            print(f'{args.head} {number+1}/15 {name} outer complete',flush=True)
    pred=pd.DataFrame(rows)
    pred.to_csv(args.output/'heldout_predictions.csv',index=False)
    pd.DataFrame(splits).to_csv(args.output/'splits.csv',index=False)
    pd.DataFrame(fits).to_csv(args.output/'fitting_metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(args.output/'training_curves.csv',index=False)
    print(json.dumps(summarize(pred,args.output,[('adapted','frozen')]),indent=2),flush=True)


if __name__=='__main__': main()
