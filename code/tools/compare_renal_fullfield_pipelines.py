"""Post-hoc paired diagnostic: fine-tuned full field versus existing pipelines.

Reuses predictions without fitting, calibration or selection of a new model.
Different token counts and heads prevent an architecture-only interpretation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_renal_2p5d_program_cache import sha
from evaluate_renal_2p5d_program import summarize

SOURCES={
    'full_adapted':('fullfield_2p5d_adaptation_v1','adapted'),
    'full_mamba':('renal_2p5d_program_mamba_v1','full'),
    'full_cox':('renal_2p5d_program_cox_v1','full'),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results_vision/renal_2p5d_pipeline_comparison_v1'))
    args=parser.parse_args()
    frames,metadata=[],{}
    reference_cohort,reference_splits=None,None
    for name,(run,model) in SOURCES.items():
        folder=Path('results_vision')/run
        cohort=pd.read_csv(folder/'cohort_common.csv').sort_values('case_id').reset_index(drop=True)
        splits=pd.read_csv(folder/'splits.csv')[['repeat','fold','case_id','partition']].sort_values(['repeat','fold','case_id']).reset_index(drop=True)
        if reference_cohort is None:
            reference_cohort,reference_splits=cohort,splits
        else:
            pd.testing.assert_frame_equal(cohort,reference_cohort)
            pd.testing.assert_frame_equal(splits,reference_splits)
        pred=pd.read_csv(folder/'heldout_predictions.csv')
        pred=pred[pred.model==model].copy()
        assert len(pred)==225 and np.isfinite(pred.risk).all()
        assert not pred.duplicated(['case_id','repeat']).any()
        labels=pred.merge(cohort,on='case_id',validate='many_to_one',suffixes=('_pred','_cohort'))
        np.testing.assert_array_equal(labels.survival_days_pred,labels.survival_days_cohort)
        np.testing.assert_array_equal(labels.event_pred,labels.event_cohort)
        pred['model']=name
        frames.append(pred)
        metadata[name]={'source':run,'source_model':model,
                        'predictions_sha256':sha(folder/'heldout_predictions.csv'),
                        'provenance_sha256':sha(folder/'provenance.json')}
    assert len(reference_cohort)==75 and int(reference_cohort.event.sum())==20
    args.output.mkdir(parents=True,exist_ok=True)
    provenance={'script_sha256':sha(__file__),'sources':metadata,'n_cases':75,'events':20,
                'analysis':'Post-hoc paired comparison of existing predictions; no refitting or new model selection',
                'limitation':'Different token counts and heads; not an architecture-only comparison',
                'hypotheses':[['full_adapted','full_mamba'],['full_adapted','full_cox']]}
    path=args.output/'provenance.json'
    if path.exists() and json.loads(path.read_text())!=provenance:
        raise ValueError('Diagnostic provenance changed; use a new output directory')
    path.write_text(json.dumps(provenance,indent=2)+'\n')
    pred=pd.concat(frames,ignore_index=True)
    pred.to_csv(args.output/'heldout_predictions.csv',index=False)
    reference_cohort.to_csv(args.output/'cohort_common.csv',index=False)
    reference_splits.to_csv(args.output/'splits.csv',index=False)
    pd.DataFrame([{'model':name,'source_run':run,'source_model':model,'reused':True}
                  for name,(run,model) in SOURCES.items()]).to_csv(args.output/'fitting_metrics.csv',index=False)
    print(json.dumps(summarize(pred,args.output,provenance['hypotheses']),indent=2))


if __name__=='__main__':
    main()
