"""Audit second-stage coverage, input pairing, early stopping and estimates."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_renal_2p5d_program import summarize
from verify_renal_2p5d_program import verify_reused_risks

RUNS={'stage2_linear_convergence_v1':2,'stage2_joint_mamba_v1':2,
      'stage2_global_local_v1':3,'stage2_dino_cox_v1':2}


def validate_curve(saved,max_epochs=100,min_epochs=20,patience=15):
    best_epoch,best_score,stale=1,-float('inf'),0
    curve=saved['curve']
    assert [r['epoch'] for r in curve]==list(range(1,len(curve)+1))
    for row in curve:
        assert np.isfinite([row['train_loss'],row['validation_cindex']]).all()
        assert 0<=row['validation_cindex']<=1
        if row['validation_cindex']>best_score+1e-8:
            best_epoch,best_score,stale=row['epoch'],row['validation_cindex'],0
        else: stale+=1
        if row['epoch']>=min_epochs and stale>=patience:
            assert row is curve[-1]
    assert saved['best_epoch']==best_epoch and saved['best_cindex']==best_score
    assert saved['epochs_run']==len(curve) and min_epochs<=len(curve)<=max_epochs
    expected='patience' if stale>=patience else 'max_epochs'
    assert saved['stop_reason']==expected
    if expected=='max_epochs': assert len(curve)==max_epochs


def main():
    root=Path('results_vision')
    baseline=root/'renal_2p5d_program_cox_v1'
    cohort_reference=pd.read_csv(baseline/'cohort_common.csv').sort_values('case_id').reset_index(drop=True)
    split_columns=['repeat','fold','case_id','partition']
    split_reference=pd.read_csv(baseline/'splits.csv')[split_columns].sort_values(['repeat','fold','case_id']).reset_index(drop=True)
    evidence={}
    for run,n_models in RUNS.items():
        folder=root/run
        if not (folder/'summary.json').exists():
            evidence[run]={'status':'incomplete'}; continue
        cohort=pd.read_csv(folder/'cohort_common.csv').sort_values('case_id').reset_index(drop=True)
        splits=pd.read_csv(folder/'splits.csv')[split_columns].sort_values(['repeat','fold','case_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(cohort,cohort_reference)
        pd.testing.assert_frame_equal(splits,split_reference)
        pred=pd.read_csv(folder/'heldout_predictions.csv')
        assert len(pred)==75*3*n_models and pred.model.nunique()==n_models and np.isfinite(pred.risk).all()
        assert not pred.duplicated(['case_id','model','repeat']).any()
        for (repeat,fold),rows in splits.groupby(['repeat','fold']):
            train=set(rows.loc[rows.partition=='train','case_id'])
            test=set(rows.loc[rows.partition=='heldout','case_id'])
            assert not train&test and train|test==set(cohort.case_id)
            for _,frame in pred[(pred.repeat==repeat)&(pred.fold==fold)].groupby('model'):
                assert set(frame.case_id)==test
        labels=pred.merge(cohort,on='case_id',validate='many_to_one',suffixes=('_pred','_cohort'))
        np.testing.assert_array_equal(labels.event_pred,labels.event_cohort)
        np.testing.assert_array_equal(labels.survival_days_pred,labels.survival_days_cohort)
        reported=json.loads((folder/'summary.json').read_text())
        comparisons=[(row['candidate'],row['reference']) for row in reported['paired_comparisons']]
        with tempfile.TemporaryDirectory(prefix='renal-stage2-audit-') as temporary:
            checked=summarize(pred,Path(temporary),comparisons)
        for name,score in reported['mean_within_fold_cindex'].items():
            assert np.isclose(score,checked['mean_within_fold_cindex'][name],atol=1e-12,rtol=0)
        for a,b in zip(reported['paired_comparisons'],checked['paired_comparisons']):
            assert a['n_bootstrap']==b['n_bootstrap']==5000
            for field in ('delta','ci95_lo','ci95_hi','bootstrap_p','p_holm_within_run'):
                assert np.isclose(a[field],b[field],atol=1e-12,rtol=0)
        result={'status':'verified','n_cases':75,'events':20,'models':n_models,'heldout_rows':len(pred),
                'paired_patient_disjoint_folds':15,'labels_match_previous_stage':True,
                'metrics_and_bootstrap_recomputed':True}
        fit=pd.read_csv(folder/'fitting_metrics.csv')
        if run in ('stage2_linear_convergence_v1','stage2_joint_mamba_v1'):
            count,capped=0,0
            for row in fit.itertuples(index=False):
                best=[]
                for inner in range(3):
                    path=folder/'folds'/f'{row.repeat}_{row.fold}_{row.model}_inner{inner}.json'
                    saved=json.loads(path.read_text()); validate_curve(saved)
                    best.append(saved['best_epoch']); count+=1; capped+=int(saved['stop_reason']=='max_epochs')
                expected=max(1,int(np.rint(np.median(best))))
                assert row.epochs==expected and json.loads(row.inner_best_epochs)==best
                saved=json.loads((folder/'folds'/f'{row.repeat}_{row.fold}_{row.model}_outer.json').read_text())
                assert saved['epoch']==expected
            assert count==90
            result['inner_curves_verified']=count
            result['inner_runs_at_cap']=capped
            result['selected_epoch_counts']={name:{str(int(k)):int(v) for k,v in frame.epochs.value_counts().sort_index().items()} for name,frame in fit.groupby('model')}
        if run=='stage2_global_local_v1':
            weights=fit[fit.model=='global_local']
            assert len(weights)==15 and np.allclose(weights.weight_full+weights.weight_local,1)
            assert weights[['weight_full','weight_local']].ge(0).all().all() and weights[['weight_full','weight_local']].le(1).all().all()
            for row in weights.itertuples(index=False):
                wide=pred[(pred.repeat==row.repeat)&(pred.fold==row.fold)].pivot(index='case_id',columns='model',values='risk')
                np.testing.assert_allclose(wide.global_local,row.weight_full*wide.full+row.weight_local*wide.local,atol=1e-12,rtol=0)
            old=pd.read_csv(root/'renal_2p5d_followup_fusion_v1'/'heldout_predictions.csv')
            result['reproduced_full_control']=verify_reused_risks(old[old.model=='full'],pred[pred.model=='full'],225)
            result['reproduced_full_control']['retrained']=True
            result['mean_local_weight']=float(weights.weight_local.mean())
            result['folds_with_zero_local_weight']=int((weights.weight_local==0).sum())
        if run=='stage2_dino_cox_v1':
            from evaluate_stage2_dino import load_arrays
            arrays=load_arrays(Path('data/embeddings/vision/stage2_dino_v1'),cohort.case_id.tolist())
            result['paired_image_sources_and_centers_verified']=True
            result['input_shapes']={name:list(x.shape) for name,x in arrays.items()}
        evidence[run]=result
    output={'runs':evidence,'all_runs_verified':all(v['status']=='verified' for v in evidence.values()),
            'scope':'Coverage, labels, paired splits, estimates, early-stop replay and selected controls; not independent clinical validation'}
    dest=root/'renal_stage2_audit'; dest.mkdir(exist_ok=True)
    (dest/'verification.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__': main()
