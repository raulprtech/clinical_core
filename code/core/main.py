"""
Multimodal Pipeline for CLINICAL-CORE / RENAL-CORE
====================================================

End-to-end orchestration of the baseline pipeline:

  TABULAR-CONN ─┐
  TEXT-CONN ────┼──→ FUSION-PROC ──→ PROGNOSIS-PROC ──→ C-index
  VISION-CONN ──┘

Driven by experiment_config.yaml. Computes the BASELINE C-index across
several modality-combination ablations:
  - Tabular only (floor)
  - Tabular + Text
  - Tabular + Vision
  - Tabular + Text + Vision (full multimodal)

This is the FIRST end-to-end measurement of CLINICAL-CORE's performance
on TCGA-KIRC. Subsequent iterations replace individual components and
re-run this pipeline to measure incremental impact.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import warnings
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from lifelines.utils import concordance_index

from components.adapters.ingestion.tabular.utils.extractor import TCGAExtractor
from components.adapters.ingestion.tabular.utils.imputation_benchmark import TabularPreprocessor
from core.registry import (
    get_imputation,
    get_text_conn,
    get_vision_conn,
    get_fusion_proc,
    get_prognosis_proc,
)


# ============================================================
# DATA DISCOVERY
# ============================================================

def discover_modality_files(
    data_dirs: Dict[str, Optional[str]],
    case_ids: List[str],
) -> pd.DataFrame:
    """
    For each case_id, find the corresponding file in each modality's directory.
    Returns DataFrame with columns: [case_id, tabular_present, text_path, vision_path].
    
    Args:
        data_dirs: dict like {'text_dir': '/path/to/reports', 'vision_dir': '/path/to/cts'}
                   None values disable that modality.
        case_ids: list of case IDs to look for.
    """
    rows = []
    text_dir = Path(data_dirs['text_dir']) if data_dirs.get('text_dir') else None
    vision_dir = Path(data_dirs['vision_dir']) if data_dirs.get('vision_dir') else None
    normalized_ids = {str(cid).upper(): cid for cid in case_ids}

    text_files_by_id = {}
    vision_volumes_by_id = {}
    dicom_dirs_by_id = {}

    def matching_case(path: Path, root: Path):
        relative = str(path.relative_to(root)).upper()
        return next((original for normalized, original in normalized_ids.items()
                     if normalized in relative), None)

    def dicom_series_root(file_path: Path, cid) -> Path:
        """Resolve ``case/series`` even when an archive adds nested folders."""
        relative_parts = file_path.relative_to(vision_dir).parts
        normalized = str(cid).upper()
        for index, part in enumerate(relative_parts):
            if normalized in part.upper() and index + 1 < len(relative_parts):
                candidate = vision_dir.joinpath(*relative_parts[:index + 2])
                if candidate.is_dir():
                    return candidate
        return file_path.parent

    if text_dir and text_dir.exists():
        for file_path in sorted(text_dir.rglob('*')):
            if file_path.is_file() and file_path.suffix.lower() in {'.pdf', '.txt', '.md'}:
                cid = matching_case(file_path, text_dir)
                if cid is not None:
                    text_files_by_id.setdefault(cid, []).append(file_path)

    if vision_dir and vision_dir.exists():
        for file_path in sorted(vision_dir.rglob('*')):
            if not file_path.is_file():
                continue
            cid = matching_case(file_path, vision_dir)
            if cid is None:
                continue
            is_nifti = file_path.suffix.lower() == '.nii' or file_path.name.lower().endswith('.nii.gz')
            if is_nifti:
                vision_volumes_by_id.setdefault(cid, []).append(file_path)
            elif (
                file_path.suffix.lower() == '.dcm'
                or (file_path.suffix == '' and file_path.name != 'series.zip')
            ):
                counts = dicom_dirs_by_id.setdefault(cid, {})
                series_dir = dicom_series_root(file_path, cid)
                counts[series_dir] = counts.get(series_dir, 0) + 1

    for cid in case_ids:
        text_path = str(text_files_by_id[cid][0]) if cid in text_files_by_id else None
        if cid in vision_volumes_by_id:
            vision_path = str(vision_volumes_by_id[cid][0])
        elif cid in dicom_dirs_by_id:
            # Prefer the directory with the largest readable series.
            vision_path = str(sorted(
                dicom_dirs_by_id[cid].items(), key=lambda item: (-item[1], str(item[0]))
            )[0][0])
        else:
            vision_path = None
        rows.append({
            'case_id': cid,
            'tabular_present': True,  # Tabular is always present (extracted from XML)
            'text_path': text_path,
            'vision_path': vision_path,
        })
    
    df = pd.DataFrame(rows).set_index('case_id')
    return df


# ============================================================
# EMBEDDING CACHE
# ============================================================

class EmbeddingCache:
    """
    Caches embeddings per modality per case to avoid recomputing across ablations.
    Stored in memory; can be persisted to disk by the runner.
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Tuple[torch.Tensor, float]]] = {}
        # Layout: {case_id: {modality: (embedding, confidence)}}
    
    def get(self, case_id: str, modality: str) -> Optional[Tuple[torch.Tensor, float]]:
        return self._cache.get(case_id, {}).get(modality)
    
    def put(self, case_id: str, modality: str, embedding: torch.Tensor, confidence: float):
        if case_id not in self._cache:
            self._cache[case_id] = {}
        self._cache[case_id][modality] = (embedding, confidence)
    
    def has(self, case_id: str, modality: str) -> bool:
        return modality in self._cache.get(case_id, {})
    
    def get_for_patient(self, case_id: str, modalities: List[str]) -> Dict[str, Tuple[torch.Tensor, float]]:
        """Returns dict of {modality: (emb, conf)} for the requested modalities, only those present."""
        out = {}
        case_data = self._cache.get(case_id, {})
        for m in modalities:
            if m in case_data:
                out[m] = case_data[m]
        return out
    
    def stats(self) -> dict:
        if not self._cache:
            return {'n_cases': 0}
        modalities_count = {}
        for case_data in self._cache.values():
            for m in case_data:
                modalities_count[m] = modalities_count.get(m, 0) + 1
        return {'n_cases': len(self._cache), 'modalities': modalities_count}


def select_cases_for_modalities(
    cache: EmbeddingCache,
    df_targets: pd.DataFrame,
    modality_subset: List[str],
) -> List[str]:
    """Select the maximal valid cohort for one modality subset.

    Unimodal experiments require only their own modality. Multimodal
    experiments use the intersection of the modalities in that combination,
    not the intersection of every modality configured in the project.
    """
    if not modality_subset:
        raise ValueError("modality_subset must contain at least one modality")

    valid_cases = []
    for cid in cache._cache:
        if cid not in df_targets.index:
            continue
        if not all(cache.has(cid, modality) for modality in modality_subset):
            continue
        survival_days = pd.to_numeric(
            pd.Series([df_targets.loc[cid, 'survival_days']]), errors='coerce'
        ).iloc[0]
        if pd.isna(survival_days) or float(survival_days) <= 0:
            continue
        valid_cases.append(cid)
    return valid_cases


# ============================================================
# MULTIMODAL PIPELINE
# ============================================================

class MultimodalPipeline:
    """
    End-to-end pipeline orchestrator.
    Uses the registry to instantiate connectors and processors by name.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.cache = EmbeddingCache()
        
        ph5 = config['phase_5_multimodal']
        self.modalities = ph5['modalities']
        self.modality_dim = ph5.get('modality_dim', 768)
        
        # Lazy initialization of connectors
        self._text_conn = None
        self._vision_conn = None
        
        # Tabular preprocessing
        self.preprocessor = TabularPreprocessor()
        self._tabular_imputation = ph5.get('tabular_imputation', 'knn_5')
        self._tabular_fitted = False
    
    def _init_text_conn(self):
        if self._text_conn is None and 'text' in self.modalities:
            text_name = self.config['phase_5_multimodal']['text_conn']
            self._text_conn = get_text_conn(text_name)
        return self._text_conn
    
    def _init_vision_conn(self):
        if self._vision_conn is None and 'vision' in self.modalities:
            ph5 = self.config['phase_5_multimodal']
            vision_name = ph5['vision_conn']
            vision_params = dict(ph5.get('vision_params', {}))
            # Preserve the legacy STU-Net setting while allowing every
            # connector parameter to remain declarative.
            vision_params.setdefault('backend', ph5.get('vision_backend', 'auto'))
            vision_params.setdefault('output_dim', self.modality_dim)
            self._vision_conn = get_vision_conn(vision_name, **vision_params)
        return self._vision_conn

    @staticmethod
    def _target_scalar(df_targets: pd.DataFrame, case_id: str, column: str) -> float:
        value = df_targets.loc[case_id, column]
        if isinstance(value, pd.Series):
            value = pd.to_numeric(value, errors='coerce').dropna()
            if value.empty:
                return np.nan
            if column == 'event':
                return float(value.max())
            return float(value.iloc[0])
        if pd.isna(value):
            return np.nan
        return float(value)
    
    def encode_cohort(
        self,
        df_features: pd.DataFrame,
        df_targets: pd.DataFrame,
        modality_files: pd.DataFrame,
        verbose: bool = True,
    ):
        """
        Run all enabled CONNs over the entire cohort and populate the cache.
        """
        # ---- TABULAR ----
        if 'tabular' in self.modalities:
            if verbose:
                print("  Encoding tabular modality...")
            
            valid_mask = df_targets['survival_days'].notna() & (df_targets['survival_days'] > 0)
            df_feat = df_features.loc[valid_mask]
            
            imp = get_imputation(self._tabular_imputation)
            X_proc, mask, conf_series = self.preprocessor.fit_transform(df_feat, imp)
            self._tabular_fitted = True
            
            # For baseline tabular CONN: zero-pad raw features to 768
            for case_id in X_proc.index:
                features = X_proc.loc[case_id].values.astype(np.float32)
                if len(features) < self.modality_dim:
                    padding = np.zeros(self.modality_dim - len(features), dtype=np.float32)
                    embedding = np.concatenate([features, padding])
                else:
                    embedding = features[:self.modality_dim]
                
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
                embedding_tensor = torch.nn.functional.normalize(
                    embedding_tensor, p=2, dim=0
                )
                self.cache.put(
                    case_id, 'tabular', embedding_tensor,
                    float(conf_series.loc[case_id])
                )
        
        # ---- TEXT ----
        if 'text' in self.modalities:
            if verbose:
                print("  Encoding text modality...")
            ph5 = self.config['phase_5_multimodal']
            embeddings_npz = ph5.get('text_embeddings_npz')
            if embeddings_npz:
                from components.adapters.ingestion.text.models.clinicalbert import (
                    load_precomputed_text_embeddings,
                )
                precomputed = load_precomputed_text_embeddings(
                    embeddings_npz, output_dim=self.modality_dim
                )
                case_lookup = {str(cid).strip().upper(): cid for cid in modality_files.index}
                for normalized_id, (emb, conf) in precomputed.items():
                    if normalized_id in case_lookup:
                        self.cache.put(case_lookup[normalized_id], 'text', emb, conf)

            text_conn = None
            for case_id, row in modality_files.iterrows():
                if self.cache.has(case_id, 'text'):
                    continue
                text_path = row.get('text_path')
                if text_path is None or pd.isna(text_path):
                    continue
                try:
                    if text_conn is None:
                        text_conn = self._init_text_conn()
                    emb, conf = text_conn.encode(text_path)
                    self.cache.put(case_id, 'text', emb, conf)
                except Exception as e:
                    warnings.warn(f"TEXT-CONN failed on {case_id}: {e}")
        
        # ---- VISION ----
        if 'vision' in self.modalities:
            if verbose:
                print("  Encoding vision modality...")
            ph5 = self.config['phase_5_multimodal']
            embeddings_csv = ph5.get('vision_embeddings_csv')
            if embeddings_csv:
                from components.adapters.ingestion.vision.models.resnet_multiview import (
                    load_precomputed_embeddings,
                )
                precomputed = load_precomputed_embeddings(
                    embeddings_csv, output_dim=self.modality_dim
                )
                case_lookup = {str(cid).strip().upper(): cid for cid in modality_files.index}
                for normalized_id, (emb, conf) in precomputed.items():
                    if normalized_id in case_lookup:
                        self.cache.put(case_lookup[normalized_id], 'vision', emb, conf)

            vision_conn = None
            for case_id, row in modality_files.iterrows():
                if self.cache.has(case_id, 'vision'):
                    continue
                vision_path = row.get('vision_path')
                if vision_path is None or pd.isna(vision_path):
                    continue
                try:
                    if vision_conn is None:
                        vision_conn = self._init_vision_conn()
                    emb, conf = vision_conn.encode(vision_path)
                    self.cache.put(case_id, 'vision', emb, conf)
                except Exception as e:
                    warnings.warn(f"VISION-CONN failed on {case_id}: {e}")
        
        if verbose:
            print(f"  Cache stats: {self.cache.stats()}")
    
    def evaluate_combination(
        self,
        modality_subset: List[str],
        df_targets: pd.DataFrame,
        seeds: List[int],
        n_folds: int = 5,
    ) -> Dict:
        """
        Evaluate the pipeline using ONLY the specified modality subset.
        Modalities not in the subset are treated as missing for all patients.
        """
        # Build fusion processor for this subset
        fusion_name = self.config['phase_5_multimodal']['fusion_proc']
        fusion = get_fusion_proc(
            fusion_name,
            modalities=modality_subset,
            modality_dims={m: self.modality_dim for m in modality_subset},
        )
        
        # Give each subset its own maximal cohort. Text-only, for example,
        # does not require tabular data; tabular+text requires only those two.
        valid_cases = select_cases_for_modalities(
            self.cache, df_targets, modality_subset,
        )
        
        if len(valid_cases) < 50:
            return {'error': f'Too few cases ({len(valid_cases)}) for cross-validation'}
        
        # Build fused embeddings for all valid cases
        fused_list = []
        targets_filtered = []
        for cid in valid_cases:
            patient_outputs = self.cache.get_for_patient(cid, modality_subset)
            fused, _ = fusion.fuse_one(patient_outputs)
            survival_days = self._target_scalar(df_targets, cid, 'survival_days')
            event = self._target_scalar(df_targets, cid, 'event')
            fused_list.append(fused)
            targets_filtered.append({
                'case_id': cid,
                'survival_days': survival_days,
                'event': int(event) if not np.isnan(event) else 0,
            })
        
        X_all = torch.stack(fused_list)
        y_df = pd.DataFrame(targets_filtered).set_index('case_id')
        
        # Cross-validated training
        seed_cis = []
        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            fold_cis = []
            
            for fold, (outer_tr_idx, test_idx) in enumerate(
                skf.split(np.zeros(len(y_df)), y_df['event'])
            ):
                # Keep the outer fold invisible during fitting and early
                # stopping. Validation is drawn only from outer-train.
                inner_tr_local, val_local = train_test_split(
                    np.arange(len(outer_tr_idx)),
                    test_size=self.config['phase_5_multimodal'].get(
                        'inner_validation_fraction', 0.20
                    ),
                    stratify=y_df.iloc[outer_tr_idx]['event'].values,
                    random_state=int(seed) * 100 + fold,
                )
                train_idx = outer_tr_idx[inner_tr_local]
                val_idx = outer_tr_idx[val_local]

                X_tr = X_all[train_idx]
                X_val = X_all[val_idx]
                X_test = X_all[test_idx]
                T_tr = torch.tensor(y_df.iloc[train_idx]['survival_days'].values, dtype=torch.float32)
                E_tr = torch.tensor(y_df.iloc[train_idx]['event'].values, dtype=torch.float32)
                T_val = torch.tensor(y_df.iloc[val_idx]['survival_days'].values, dtype=torch.float32)
                E_val = torch.tensor(y_df.iloc[val_idx]['event'].values, dtype=torch.float32)
                
                prognosis_name = self.config['phase_5_multimodal']['prognosis_proc']
                prognosis = get_prognosis_proc(
                    prognosis_name,
                    fused_dim=fusion.fused_dim,
                )
                result = prognosis.fit(
                    X_tr, T_tr, E_tr, X_val, T_val, E_val,
                    epochs=self.config['phase_5_multimodal'].get('prognosis_epochs', 200),
                    patience=self.config['phase_5_multimodal'].get('prognosis_patience', 20),
                    verbose=False,
                )
                test_risk = prognosis.predict_risk(X_test)
                test_ci = concordance_index(
                    y_df.iloc[test_idx]['survival_days'].values,
                    -test_risk,
                    y_df.iloc[test_idx]['event'].values,
                )
                fold_cis.append(float(test_ci))
            
            seed_cis.append(float(np.mean(fold_cis)))
        
        return {
            'modalities': modality_subset,
            'n_cases': len(valid_cases),
            'cindex_mean': float(np.mean(seed_cis)),
            'cindex_std': float(np.std(seed_cis)),
            'cindex_per_seed': seed_cis,
        }
    
    def run_ablation(
        self,
        df_features: pd.DataFrame,
        df_targets: pd.DataFrame,
        modality_files: pd.DataFrame,
        seeds: List[int],
        n_folds: int = 5,
        ablations: Optional[List[List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Full ablation: encode cohort once, then evaluate multiple modality subsets.
        """
        phase_cfg = self.config['phase_5_multimodal']
        context_cfg = self.config.get('clinical_context', {})
        clinical_moment = phase_cfg.get(
            'clinical_moment', context_cfg.get('moment', 'post_surgery')
        )
        if clinical_moment not in {'pre_surgery', 'post_surgery'}:
            raise ValueError(
                "phase_5_multimodal.clinical_moment must be "
                "'pre_surgery' or 'post_surgery'"
            )
        pathology_modalities = set(
            phase_cfg.get(
                'pathology_modalities',
                context_cfg.get('pathology_modalities', ['text']),
            )
        )

        # 1. Encode cohort once (cached)
        self.encode_cohort(df_features, df_targets, modality_files)
        
        # 2. Default ablations: each unimodal cohort, then pairs and full.
        if ablations is None:
            ablations = []
            if 'tabular' in self.modalities:
                ablations.append(['tabular'])
            if 'text' in self.modalities:
                ablations.append(['text'])
            if 'vision' in self.modalities:
                ablations.append(['vision'])
            if 'tabular' in self.modalities and 'text' in self.modalities:
                ablations.append(['tabular', 'text'])
            if 'tabular' in self.modalities and 'vision' in self.modalities:
                ablations.append(['tabular', 'vision'])
            if 'text' in self.modalities and 'vision' in self.modalities:
                ablations.append(['text', 'vision'])
            if all(m in self.modalities for m in ['tabular', 'text', 'vision']):
                ablations.append(['tabular', 'text', 'vision'])
        
        results = []
        for subset in ablations:
            if (
                clinical_moment == 'pre_surgery'
                and pathology_modalities.intersection(subset)
            ):
                print(
                    f"  Skipping subset {subset}: pathology is unavailable "
                    "at the pre-surgery evaluation moment"
                )
                continue
            print(f"  Evaluating subset: {subset}")
            r = self.evaluate_combination(subset, df_targets, seeds, n_folds)
            r['subset_label'] = '+'.join(subset)
            r['clinical_moment'] = clinical_moment
            results.append(r)
            if 'error' not in r:
                print(f"    C-index: {r['cindex_mean']:.4f} ± {r['cindex_std']:.4f} (n={r['n_cases']})")
            else:
                print(f"    ERROR: {r['error']}")
        
        return pd.DataFrame(results)
