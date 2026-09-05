"""Repeated nested renal-region experiments with per-fold durable outputs.

Primary score averages within-fold C-indices, avoiding cross-model risk-scale
comparisons. Patient bootstrap resamples a patient jointly across repetitions.
Frozen Mamba and compact Cox/radiomics experiments have separate result folders.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis

from build_renal_2p5d_program_cache import ARMS, sha
from evaluate_resnet_sequence_models import load_sequences, load_targets, pad_sequences, safe_cindex
from evaluate_resnet_sequence_nested_cv import select_epochs_inner, refit_and_predict


def comparable_scores(times, events, risks):
    comparable = events[:, None].astype(bool) & (
        (times[:, None] < times[None, :]) |
        ((times[:, None] == times[None, :]) & ~events[None, :].astype(bool)))
    diff = risks[:, None] - risks[None, :]
    score = (diff > 0).astype(float) + .5 * (diff == 0)
    return comparable.astype(float), score * comparable


def summarize(predictions, output, comparisons, iterations=5000):
    names = list(predictions.model.unique())
    ids = sorted(predictions.case_id.unique())
    id_to_index = {case: i for i, case in enumerate(ids)}
    rng = np.random.default_rng(54049)
    weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=iterations)
    observed, samples, rows = {}, {}, []
    for name in names:
        scores, boot = [], []
        for (repeat, fold), frame in predictions[predictions.model == name].groupby(['repeat', 'fold']):
            if frame.case_id.duplicated().any():
                raise ValueError('Duplicate held-out patients')
            t, e, risk = frame.survival_days.to_numpy(), frame.event.to_numpy(), frame.risk.to_numpy()
            denom, numer = comparable_scores(t, e, risk)
            score = numer.sum() / denom.sum()
            if not np.isclose(score, safe_cindex(t, risk, e)):
                raise ValueError('Pair matrix disagrees with validated C-index')
            local_w = weights[:, [id_to_index[x] for x in frame.case_id]]
            d = np.einsum('bi,ij,bj->b', local_w, denom, local_w, optimize=True)
            n = np.einsum('bi,ij,bj->b', local_w, numer, local_w, optimize=True)
            boot.append(np.divide(n, d, out=np.full_like(n, np.nan), where=d > 0))
            scores.append(score)
            rows.append({'model': name, 'repeat': repeat, 'fold': fold,
                         'n': len(frame), 'events': int(e.sum()), 'cindex': score})
        observed[name], samples[name] = float(np.mean(scores)), np.nanmean(boot, axis=0)
    result = []
    for candidate, reference in comparisons:
        delta = samples[candidate] - samples[reference]
        delta = delta[np.isfinite(delta)]
        low, high = np.quantile(delta, [.025, .975])
        p = min(1., 2 * min(np.mean(delta <= 0), np.mean(delta >= 0)))
        result.append({'candidate': candidate, 'reference': reference,
                       'delta': observed[candidate] - observed[reference],
                       'ci95_lo': float(low), 'ci95_hi': float(high), 'bootstrap_p': p,
                       'n_bootstrap': len(delta)})
    order = np.argsort([r['bootstrap_p'] for r in result])
    adjusted = 0.
    for rank, i in enumerate(order):
        adjusted = max(adjusted, min(1., (len(order)-rank) * result[i]['bootstrap_p']))
        result[i]['p_holm_within_run'] = adjusted
    pd.DataFrame(rows).to_csv(output / 'per_fold_metrics.csv', index=False)
    pd.DataFrame(rows).groupby(['model', 'repeat'], as_index=False).agg(
        mean_fold_cindex=('cindex', 'mean')).to_csv(output / 'per_repeat_metrics.csv', index=False)
    pd.DataFrame(result).to_csv(output / 'paired_cluster_bootstrap.csv', index=False)
    summary = {'mean_within_fold_cindex': observed, 'paired_comparisons': result,
               'primary_estimand': 'Arithmetic mean of within-outer-fold C-indices',
               'uncertainty': 'Patient resampling across all repeated folds; fitted predictions fixed',
               'limitation': 'Exploratory reused cohort; bootstrap does not capture retraining uncertainty'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def cox_nested(x, times, events, train, heldout, seed):
    y = np.array(list(zip(events.astype(bool), times)), dtype=[('event', '?'), ('time', '<f8')])
    best = None
    splits = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(train, events[train]))
    for components in (4, 8):
        for alpha in (100., 10., 1.):
            scores = []
            for tr, va in splits:
                pipeline = make_pipeline(StandardScaler(), PCA(components, svd_solver='full'),
                                         StandardScaler(), CoxPHSurvivalAnalysis(alpha=alpha, n_iter=200))
                pipeline.fit(x[train[tr]], y[train[tr]])
                scores.append(safe_cindex(times[train[va]], pipeline.predict(x[train[va]]), events[train[va]]))
            score = float(np.mean(scores))
            if best is None or score > best[0]:
                best = (score, components, alpha)
    score, components, alpha = best
    pipeline = make_pipeline(StandardScaler(), PCA(components, svd_solver='full'),
                             StandardScaler(), CoxPHSurvivalAnalysis(alpha=alpha, n_iter=200))
    pipeline.fit(x[train], y[train])
    return pipeline.predict(x[heldout]), {'components': components, 'alpha': alpha, 'inner_cindex': score}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=['mamba', 'cox'], required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--outer-repeats', type=int, default=3)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--bootstrap-iterations', type=int, default=5000)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.use_position = False
    args.inner_folds, args.patience = 3, 20
    args.lr, args.weight_decay, args.dropout = .001, .001, .1
    args.model_dim, args.attention_dim, args.state_dim, args.mamba_blocks = 128, 64, 16, 2
    sequences = {arm: load_sequences(args.cache / arm) for arm in ARMS}
    targets = load_targets(args.targets)
    radiomics = pd.read_csv(args.cache / 'radiomics_2d.csv').set_index('case_id')
    common = sorted(set.intersection(set(targets.index), set(radiomics.index),
                                    *[set(x) for x in sequences.values()]))
    cohort = targets.loc[common]
    cohort = cohort[(cohort.survival_days > 0) & cohort.event.isin([0, 1])]
    ids = cohort.index.tolist()
    times, events = cohort.survival_days.to_numpy(float), cohort.event.to_numpy(int)
    if len(ids) < 25 or events.sum() < 10:
        raise ValueError('Too few eligible patients/events for this protocol')
    if args.kind == 'mamba':
        tensors = {arm: pad_sequences(seq, ids) for arm, seq in sequences.items()}
        models = list(ARMS)
        comparisons = [('renal_crop', 'full'), ('renal_slices', 'full'), ('renal_crop', 'renal_slices')]
    else:
        arrays = {arm: np.stack([sequences[arm][case][0].mean(axis=0) for case in ids]) for arm in ARMS}
        arrays['radiomics'] = radiomics.loc[ids].to_numpy(float)
        models = list(arrays)
        comparisons = [('renal_crop', 'full'), ('renal_slices', 'full'), ('radiomics', 'renal_crop')]
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / 'folds'
    checkpoints.mkdir(exist_ok=True)
    provenance = {'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  'script_sha256': sha(__file__), 'targets_sha256': sha(args.targets),
                  'cache_contract_sha256': sha(args.cache / 'contract.json'),
                  'radiomics_sha256': sha(args.cache / 'radiomics_2d.csv'),
                  'ids_sha256': __import__('hashlib').sha256('\n'.join(ids).encode()).hexdigest(),
                  'n_cases': len(ids), 'events': int(events.sum())}
    provenance_path = args.output / 'provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError('Run provenance changed; choose a new result directory')
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    cohort.reset_index().to_csv(args.output / 'cohort_common.csv', index=False)
    split_plan, predictions, fits = [], [], []
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.outer_repeats, random_state=4049)
    start = time.monotonic()
    for number, (train, heldout) in enumerate(splitter.split(np.zeros(len(ids)), events)):
        repeat, fold = divmod(number, 5)
        seed = 4049 + repeat * 100 + fold
        split_plan.extend({'repeat': repeat, 'fold': fold, 'case_id': ids[i],
                           'partition': 'train' if i in train else 'heldout'} for i in range(len(ids)))
        for name in models:
            path = checkpoints / f'{repeat}_{fold}_{name}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['heldout_ids'] != [ids[i] for i in heldout]:
                    raise ValueError('Checkpoint patient mismatch')
                risk, fit = np.array(saved['risks']), saved['fit']
            else:
                if args.kind == 'mamba':
                    f, pos, mask = tensors[name]
                    selected, inner_epochs, scores = select_epochs_inner(
                        'mamba', f, pos, mask, times, events, train, seed, args, torch.device(args.device))
                    risk, params = refit_and_predict('mamba', f, pos, mask, times, events, train,
                                                     heldout, selected, seed * 1000 + 29, args, torch.device(args.device))
                    fit = {'epochs': selected, 'inner_epochs': inner_epochs,
                           'inner_cindex': float(np.mean(scores)), 'parameters': params}
                else:
                    risk, fit = cox_nested(arrays[name], times, events, train, heldout, seed)
                if not np.isfinite(risk).all():
                    raise ValueError('Nonfinite held-out predictions')
                saved = {'heldout_ids': [ids[i] for i in heldout], 'risks': risk.tolist(), 'fit': fit}
                partial = path.with_suffix('.partial')
                partial.write_text(json.dumps(saved))
                partial.replace(path)
            predictions.extend({'repeat': repeat, 'fold': fold, 'model': name, 'case_id': ids[i],
                                'survival_days': float(times[i]), 'event': int(events[i]), 'risk': float(r)}
                               for i, r in zip(heldout, risk))
            fits.append({'repeat': repeat, 'fold': fold, 'model': name, **fit})
            print(f'{number+1}/15 {name} complete ({time.monotonic()-start:.0f}s)', flush=True)
    pred = pd.DataFrame(predictions)
    pred.to_csv(args.output / 'heldout_predictions.csv', index=False)
    pd.DataFrame(split_plan).to_csv(args.output / 'splits.csv', index=False)
    pd.DataFrame(fits).to_csv(args.output / 'fitting_metrics.csv', index=False)
    summary = summarize(pred, args.output, comparisons, args.bootstrap_iterations)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
