"""
Diagnostico: Cox directo sobre las 19 variables crudas con holdout 80/20.

Proposito: aislar si el gap CV-in-pool vs held-out (~0.18 puntos) viene del
pipeline multi-etapa encoder -> VAE -> Cox, o si es propiedad intrinseca del
dataset TCGA-KIRC en este tamano de holdout.

Referencia comparativa:
- Phase 2 Cox baseline (CV 5x5 sobre 444 pacientes): 0.798 +/- 0.041
- Phase 8 Cox (pipeline completo):
    in-pool CV    = 0.8631
    held-out 80/20 = 0.6801   <- el numero a contrastar

Interpretacion:
- Si Cox crudo held-out >= 0.75: el pipeline encoder-VAE es el culpable.
- Si Cox crudo held-out ~ 0.68: el gap es intrinseco al dataset, 0.68 es real.
- Si Cox crudo held-out < 0.68: caso patologico, revisar particion.

Requisitos:
    pip install lifelines scikit-learn pandas numpy --break-system-packages

Uso:
    python diagnostic_cox_raw.py \\
        --features raw_features.csv \\
        --targets  raw_targets.csv
"""

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import sklearn
import lifelines


def load_data(features_path: str, targets_path: str):
    """Carga y filtra el cohorte como lo hace el pipeline principal."""
    feats = pd.read_csv(features_path)
    tgts = pd.read_csv(targets_path)

    # Merge por case_id
    df = feats.merge(tgts, on='case_id', how='inner')

    # Filtro anti-leakage de tiempo (bug 2 ya aplicado upstream, pero por si acaso)
    df = df[(df['survival_days'] > 0) & df['survival_days'].notna()].reset_index(drop=True)

    # Columnas feature (excluir case_id + targets)
    feature_cols = [c for c in feats.columns if c != 'case_id']
    X = df[feature_cols].values
    T = df['survival_days'].values.astype(float)
    E = df['event'].values.astype(int)

    print(f"Cohorte valido: n={len(df)}, eventos={E.sum()}, "
          f"censura={(1-E.mean()):.1%}, mediana seguimiento={np.median(T):.0f} dias")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print()

    return X, T, E, feature_cols


def fit_cox_and_score(X_tr, T_tr, E_tr, X_va, T_va, E_va, feature_cols,
                      penalizer: float = 0.5):
    """Ajusta Cox con imputacion KNN y standardizacion en train, transforma holdout.

    Defensas para convergencia:
    - Descarta columnas con >90% missing en train (no hay senal suficiente)
    - Descarta columnas con varianza casi cero tras imputacion
    - Fallback escalado de penalizer si Newton-Raphson diverge
    """
    X_tr = np.asarray(X_tr, dtype=float)
    X_va = np.asarray(X_va, dtype=float)

    # Drop cols con >90% missing en train
    miss_frac = np.isnan(X_tr).mean(axis=0)
    keep_mask = miss_frac < 0.90
    X_tr = X_tr[:, keep_mask]
    X_va = X_va[:, keep_mask]
    kept_cols = [c for c, k in zip(feature_cols, keep_mask) if k]

    # Imputacion KNN fit en train
    imp = KNNImputer(n_neighbors=5)
    X_tr_imp = imp.fit_transform(X_tr)
    X_va_imp = imp.transform(X_va)

    # Drop cols con varianza casi cero tras imputacion (columnas constantes)
    var_tr = X_tr_imp.var(axis=0)
    var_mask = var_tr > 1e-8
    X_tr_imp = X_tr_imp[:, var_mask]
    X_va_imp = X_va_imp[:, var_mask]
    kept_cols = [c for c, k in zip(kept_cols, var_mask) if k]

    # Standardize
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_imp)
    X_va_s = sc.transform(X_va_imp)

    # DataFrames para lifelines
    df_tr = pd.DataFrame(X_tr_s, columns=kept_cols)
    df_tr['T'] = T_tr
    df_tr['E'] = E_tr

    # Cox con escalado de penalizer si diverge
    last_err = None
    cph = None
    for pen_try in [penalizer, 1.0, 5.0, 20.0]:
        try:
            cph = CoxPHFitter(penalizer=pen_try, l1_ratio=0.0)
            cph.fit(df_tr, duration_col='T', event_col='E', show_progress=False)
            last_err = None
            break
        except Exception as ex:
            last_err = ex
            cph = None
    if cph is None:
        raise RuntimeError(f"Cox no convergio ni con penalizer=20.0: {last_err}")

    # Scores = partial hazard
    df_va = pd.DataFrame(X_va_s, columns=kept_cols)
    risk_va = cph.predict_partial_hazard(df_va).values
    risk_tr = cph.predict_partial_hazard(df_tr.drop(columns=['T', 'E'])).values

    # C-index (mayor riesgo -> menor supervivencia esperada)
    # concordance_index de lifelines espera: event_times, predicted_scores, event_observed
    # Pero predicted_scores debe ser tal que valores mayores = riesgo mayor = supervivencia menor
    # Entonces pasamos -risk para que sea consistente con "tiempo esperado"
    ci_tr = concordance_index(T_tr, -risk_tr, E_tr)
    ci_va = concordance_index(T_va, -risk_va, E_va)
    return ci_tr, ci_va


def run_cv_5x5(X, T, E, feature_cols, seeds=(42, 123, 456, 789, 1024),
               n_folds=5, penalizer=0.1):
    """CV 5x5 igual al protocolo Phase 2 para comparar manzanas con manzanas."""
    print(f"=== Cox crudo: CV {n_folds}-folds x {len(seeds)} semillas ===")
    seed_means = []
    detail_rows = []
    for s in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s)
        fold_cis = []
        for fold_idx, (tr_idx, va_idx) in enumerate(
                skf.split(np.zeros(len(E)), E)):
            _, ci_va = fit_cox_and_score(
                X[tr_idx], T[tr_idx], E[tr_idx],
                X[va_idx], T[va_idx], E[va_idx],
                feature_cols, penalizer=penalizer,
            )
            fold_cis.append(ci_va)
            detail_rows.append({
                'evaluation': 'cv_in_pool',
                'seed': int(s), 'fold': int(fold_idx),
                'cindex': float(ci_va),
                'n_train': int(len(tr_idx)), 'n_val': int(len(va_idx)),
                'events_val': int(E[va_idx].sum()),
            })
        seed_mean = float(np.mean(fold_cis))
        seed_means.append(seed_mean)
        print(f"  seed={s:<5d}  mean={seed_mean:.4f}  folds={[f'{x:.4f}' for x in fold_cis]}")
    arr = np.array(seed_means)
    print(f"\n  CV agregado: mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"median={np.median(arr):.4f}  [{arr.min():.4f}, {arr.max():.4f}]")
    return arr, detail_rows


def run_holdout(X, T, E, feature_cols, seeds=(42, 123, 456, 789, 1024),
                holdout_frac=0.20, penalizer=0.1):
    """Holdout 80/20 estratificado por evento, una particion por semilla."""
    print(f"\n=== Cox crudo: holdout {int((1-holdout_frac)*100)}/{int(holdout_frac*100)} "
          f"estratificado x {len(seeds)} semillas ===")
    ho_cis = []
    tr_cis = []
    detail_rows = []
    for s in seeds:
        tr_idx, ho_idx = train_test_split(
            np.arange(len(E)), test_size=holdout_frac,
            stratify=E, random_state=s,
        )
        ci_tr, ci_ho = fit_cox_and_score(
            X[tr_idx], T[tr_idx], E[tr_idx],
            X[ho_idx], T[ho_idx], E[ho_idx],
            feature_cols, penalizer=penalizer,
        )
        tr_cis.append(ci_tr)
        ho_cis.append(ci_ho)
        detail_rows.append({
            'evaluation': 'holdout',
            'seed': int(s),
            'cindex_train': float(ci_tr),
            'cindex_holdout': float(ci_ho),
            'n_train': int(len(tr_idx)), 'n_holdout': int(len(ho_idx)),
            'events_holdout': int(E[ho_idx].sum()),
        })
        print(f"  seed={s:<5d}  train={ci_tr:.4f}  "
              f"held-out={ci_ho:.4f}  (n_ho={len(ho_idx)}, eventos_ho={E[ho_idx].sum()})")

    arr_ho = np.array(ho_cis)
    arr_tr = np.array(tr_cis)
    print(f"\n  Train agregado:    mean={arr_tr.mean():.4f}  std={arr_tr.std():.4f}")
    print(f"  Held-out agregado: mean={arr_ho.mean():.4f}  std={arr_ho.std():.4f}  "
          f"median={np.median(arr_ho):.4f}  [{arr_ho.min():.4f}, {arr_ho.max():.4f}]")
    print(f"  Gap train - held-out: {arr_tr.mean() - arr_ho.mean():+.4f}")
    return arr_tr, arr_ho, detail_rows


def diagnose(cv_arr, ho_arr):
    """Interpretacion final del diagnostico."""
    print("\n" + "="*70)
    print("DIAGNOSTICO")
    print("="*70)
    cv_m = cv_arr.mean()
    ho_m = ho_arr.mean()
    gap = cv_m - ho_m

    pipeline_cv_inpool = 0.8631
    pipeline_holdout = 0.6801
    pipeline_gap = pipeline_cv_inpool - pipeline_holdout

    print(f"\n  Cox crudo  CV in-pool (5x5):   {cv_m:.4f}")
    print(f"  Cox crudo  held-out (80/20):   {ho_m:.4f}")
    print(f"  Cox crudo  gap CV - held-out:  {gap:+.4f}")
    print()
    print(f"  Pipeline   CV in-pool:         {pipeline_cv_inpool:.4f}")
    print(f"  Pipeline   held-out:           {pipeline_holdout:.4f}")
    print(f"  Pipeline   gap CV - held-out:  {pipeline_gap:+.4f}")
    print()

    print("  INTERPRETACION:")
    if ho_m >= 0.75:
        print(f"  >>> El pipeline encoder->VAE es el CULPABLE del overfit. <<<")
        print(f"  Cox crudo held-out ({ho_m:.3f}) es significativamente mejor")
        print(f"  que el pipeline ({pipeline_holdout:.3f}). La cadena multi-etapa")
        print(f"  anade sobreajuste que el Cox simple no sufre.")
        print(f"  Accion recomendada: simplificar pipeline o entrenar end-to-end.")
    elif abs(ho_m - pipeline_holdout) <= 0.03:
        print(f"  >>> El gap es INTRINSECO al dataset. <<<")
        print(f"  Cox crudo held-out ({ho_m:.3f}) esta cerca del pipeline")
        print(f"  ({pipeline_holdout:.3f}). El dataset TCGA-KIRC con este tamano")
        print(f"  de holdout no genera mejor C-index aunque el modelo sea simple.")
        print(f"  El 0.68 es el numero real para Paper 2. Unica palanca: mas datos")
        print(f"  (modalidades reales, cohorte externo) o K-fold externo para")
        print(f"  mejor estimador de varianza.")
    else:
        print(f"  >>> Caso intermedio. <<<")
        print(f"  Parte del gap viene del pipeline, parte del dataset.")
        print(f"  Delta pipeline-vs-crudo en holdout: {ho_m - pipeline_holdout:+.4f}")


def main():
    ap = argparse.ArgumentParser(
        description='Diagnostico Cox crudo sobre 19 features (CLINICAL-CORE)')
    ap.add_argument('--features', required=True,
                    help='Ruta a raw_features.csv')
    ap.add_argument('--targets', required=True,
                    help='Ruta a raw_targets.csv')
    ap.add_argument('--output-dir', default='./diagnostic_results',
                    help='Directorio donde guardar los CSV y JSON de salida')
    ap.add_argument('--penalizer', type=float, default=0.5,
                    help='L2 penalty for CoxPH (default 0.5)')
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[42, 123, 456, 789, 1024],
                    help='Semillas (default: 42 123 456 789 1024)')
    args = ap.parse_args()

    # Setup output dir
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output_dir) / f"diagnostic_cox_raw_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}")
    print(f"Output dir: {out_dir.resolve()}\n")

    # Environment metadata (trazabilidad para comparar con mi corrida)
    env_info = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scikit_learn': sklearn.__version__,
        'lifelines': lifelines.__version__,
        'args': {
            'features': str(Path(args.features).resolve()),
            'targets': str(Path(args.targets).resolve()),
            'penalizer': args.penalizer,
            'seeds': list(args.seeds),
        },
    }
    print("Environment:")
    for k, v in env_info.items():
        if k != 'args':
            print(f"  {k}: {v}")
    print()

    # Run
    X, T, E, feature_cols = load_data(args.features, args.targets)
    cv_arr, cv_rows = run_cv_5x5(X, T, E, feature_cols, seeds=args.seeds,
                                 penalizer=args.penalizer)
    tr_arr, ho_arr, ho_rows = run_holdout(X, T, E, feature_cols,
                                           seeds=args.seeds,
                                           penalizer=args.penalizer)
    diagnose(cv_arr, ho_arr)

    # Persist detailed results
    cv_df = pd.DataFrame(cv_rows)
    ho_df = pd.DataFrame(ho_rows)
    cv_df.to_csv(out_dir / 'cv_in_pool_folds.csv', index=False)
    ho_df.to_csv(out_dir / 'holdout_seeds.csv', index=False)

    summary = {
        'environment': env_info,
        'cohort': {
            'n_cases': int(len(E)),
            'n_events': int(E.sum()),
            'censoring_rate': float(1 - E.mean()),
            'median_followup_days': float(np.median(T)),
            'n_features_input': int(len(feature_cols)),
            'feature_names': list(feature_cols),
        },
        'results': {
            'cv_in_pool': {
                'seed_means': [float(x) for x in cv_arr],
                'aggregate_mean': float(cv_arr.mean()),
                'aggregate_std': float(cv_arr.std()),
                'aggregate_median': float(np.median(cv_arr)),
            },
            'holdout': {
                'train_seed_cis': [float(x) for x in tr_arr],
                'holdout_seed_cis': [float(x) for x in ho_arr],
                'train_mean': float(tr_arr.mean()),
                'holdout_mean': float(ho_arr.mean()),
                'holdout_std': float(ho_arr.std()),
                'holdout_median': float(np.median(ho_arr)),
                'gap_train_holdout': float(tr_arr.mean() - ho_arr.mean()),
            },
            'pipeline_reference_for_comparison': {
                'cv_in_pool': 0.8631,
                'holdout': 0.6801,
                'source': 'phase_8_prognosis run 20260422_174513_5dbfc227',
            },
        },
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nArchivos guardados:")
    print(f"  {out_dir / 'cv_in_pool_folds.csv'}   ({len(cv_df)} filas)")
    print(f"  {out_dir / 'holdout_seeds.csv'}      ({len(ho_df)} filas)")
    print(f"  {out_dir / 'summary.json'}")
    print(f"\nComparte el contenido de {out_dir.name}/ para verificar los numeros.")


if __name__ == '__main__':
    main()