"""Audit completed experiment coverage, paired splits and aggregate claims."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_resnet_sequence_models import safe_cindex

RUNS = {
    'renal_2p5d_program_cox_v1': 4,
    'renal_2p5d_program_mamba_v1': 3,
    'renal_2p5d_adaptation_v1': 2,
    'renal_2p5d_followup_moments_v1': 4,
    'renal_2p5d_followup_fusion_v1': 5,
    'renal_2p5d_followup_fusion_moments_v1': 5,
    'fullfield_moments_214_v1': 2,
    'renal_2p5d_adaptation_extended_v1': 2,
}


def main():
    root = Path('results_vision')
    evidence, frames, errors = {}, {}, []
    reference_splits = {}
    for run, n_models in RUNS.items():
        folder = root/run
        if not (folder/'summary.json').exists():
            evidence[run] = {'status': 'incomplete'}
            continue
        pred = pd.read_csv(folder/'heldout_predictions.csv')
        splits = pd.read_csv(folder/'splits.csv')
        cohort = pd.read_csv(folder/'cohort_common.csv')
        frames[run] = pred
        expected = set(cohort.case_id)
        assert len(cohort) == (214 if run == 'fullfield_moments_214_v1' else 75)
        assert not cohort.case_id.duplicated().any()
        assert pred.model.nunique() == n_models
        assert np.isfinite(pred.risk).all()
        assert len(pred) == len(cohort)*3*n_models
        assert not pred.duplicated(['case_id','model','repeat']).any()
        ordered = splits[['repeat','fold','case_id','partition']].sort_values(['repeat','fold','case_id']).reset_index(drop=True)
        cohort_key = len(cohort)
        if cohort_key not in reference_splits:
            reference_splits[cohort_key] = ordered
        else:
            pd.testing.assert_frame_equal(ordered, reference_splits[cohort_key])
        for (repeat,fold), s in splits.groupby(['repeat','fold']):
            train = set(s.loc[s.partition=='train','case_id'])
            test = set(s.loc[s.partition=='heldout','case_id'])
            assert not train & test and train|test == expected
            for _, f in pred[(pred.repeat==repeat)&(pred.fold==fold)].groupby('model'):
                assert set(f.case_id) == test
        summary = json.loads((folder/'summary.json').read_text())
        for name, reported in summary['mean_within_fold_cindex'].items():
            scores = [safe_cindex(f.survival_days.to_numpy(), f.risk.to_numpy(), f.event.to_numpy())
                      for _,f in pred[pred.model==name].groupby(['repeat','fold'])]
            assert len(scores) == 15
            assert np.isclose(reported, np.mean(scores), atol=1e-12, rtol=0)
        for row in summary['paired_comparisons']:
            actual = summary['mean_within_fold_cindex'][row['candidate']] - summary['mean_within_fold_cindex'][row['reference']]
            assert np.isclose(row['delta'],actual,atol=1e-12,rtol=0)
            assert row['n_bootstrap'] == 5000 and row['ci95_lo']<=row['ci95_hi']
        evidence[run] = {'status':'verified', 'n_cases':len(cohort), 'events':int(cohort.event.sum()),
                         'models':n_models, 'heldout_rows':len(pred), 'paired_outer_folds':15,
                         'patient_disjoint_train_test':True, 'aggregate_metrics_recomputed':True}
    if 'renal_2p5d_program_cox_v1' in frames and 'renal_2p5d_followup_moments_v1' in frames:
        left = frames['renal_2p5d_program_cox_v1']
        left = left[left.model.isin(['full','renal_crop'])]
        right = frames['renal_2p5d_followup_moments_v1']
        right = right[right.model.isin(['full','renal_crop'])]
        keys = ['case_id','model','repeat','fold']
        merged = left.merge(right,on=keys,validate='one_to_one')
        assert len(merged) == len(left) == len(right)
        difference = float(np.max(np.abs(merged.risk_x-merged.risk_y)))
        assert difference == 0
        evidence['reproduced_controls'] = {'predictions':len(merged),'max_absolute_difference':difference}
    if 'renal_2p5d_followup_fusion_v1' in frames:
        fit = pd.read_csv(root/'renal_2p5d_followup_fusion_v1'/'fitting_metrics.csv')
        weights = fit[['weight_tabular','weight_other']].dropna().to_numpy()
        assert len(weights)==30 and np.allclose(weights.sum(axis=1),1)
        assert ((weights>=0)&(weights<=1)).all()
        evidence['fusion_weights']={'pairs':len(weights),'convex':True}
    result={'runs':evidence,'all_runs_verified':all(evidence[r]['status']=='verified' for r in RUNS),
            'scope':'Coverage, split pairing, aggregate recomputation and control reproduction; code leakage review and anatomical validity are separate checks'}
    dest=root/'renal_2p5d_program_audit'
    dest.mkdir(exist_ok=True)
    (dest/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
