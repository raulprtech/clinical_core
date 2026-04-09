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
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index

from extractor import TCGAExtractor
from imputation_benchmark import TabularPreprocessor
from registry import (
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
    
    # Pre-index files for fast lookup
    text_files_by_id = {}
    if text_dir and text_dir.exists():
        for f in text_dir.rglob('*'):
            if f.is_file() and f.suffix.lower() in ['.pdf', '.txt', '.md']:
                for cid in case_ids:
                    if cid in f.name:
                        text_files_by_id.setdefault(cid, []).append(f)
    
    vision_files_by_id = {}
    if vision_dir and vision_dir.exists():
        for f in vision_dir.rglob('*'):
            if f.is_file() and (f.suffix.lower() == '.gz' or f.suffix.lower() in ['.nii', '.dcm']):
                for cid in case_ids:
                    if cid in f.name:
                        vision_files_by_id.setdefault(cid, []).append(f)
    
    for cid in case_ids:
        text_path = str(text_files_by_id[cid][0]) if cid in text_files_by_id else None
        vision_path = str(vision_files_by_id[cid][0]) if cid in vision_files_by_id else None
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
            backend = ph5.get('vision_backend', 'auto')
            self._vision_conn = get_vision_conn(vision_name, backend=backend)
        return self._vision_conn
    
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
                
                # L2 normalize
                norm = np.linalg.norm(embedding)
                if norm > 1e-8:
                    embedding = embedding / norm
                
                self.cache.put(
                    case_id, 'tabular',
                    torch.tensor(embedding, dtype=torch.float32),
                    float(conf_series.loc[case_id])
                )
        
        # ---- TEXT ----
        if 'text' in self.modalities:
            if verbose:
                print("  Encoding text modality...")
            text_conn = self._init_text_conn()
            
            for case_id, row in modality_files.iterrows():
                text_path = row.get('text_path')
                if text_path is None:
                    continue
                try:
                    emb, conf = text_conn.encode(text_path)
                    self.cache.put(case_id, 'text', emb, conf)
                except Exception as e:
                    warnings.warn(f"TEXT-CONN failed on {case_id}: {e}")
        
        # ---- VISION ----
        if 'vision' in self.modalities:
            if verbose:
                print("  Encoding vision modality...")
            vision_conn = self._init_vision_conn()
            
            for case_id, row in modality_files.iterrows():
                vision_path = row.get('vision_path')
                if vision_path is None:
                    continue
                try:
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
        
        # Filter cases that have at least the tabular modality
        valid_cases = [
            cid for cid in self.cache._cache.keys()
            if 'tabular' in self.cache._cache[cid]
            and cid in df_targets.index
        ]
        valid_cases = [cid for cid in valid_cases
                       if df_targets.loc[cid, 'survival_days'] > 0]
        
        if len(valid_cases) < 50:
            return {'error': f'Too few cases ({len(valid_cases)}) for cross-validation'}
        
        # Build fused embeddings for all valid cases
        fused_list = []
        targets_filtered = []
        for cid in valid_cases:
            patient_outputs = self.cache.get_for_patient(cid, modality_subset)
            fused, _ = fusion.fuse_one(patient_outputs)
            fused_list.append(fused)
            targets_filtered.append({
                'case_id': cid,
                'survival_days': df_targets.loc[cid, 'survival_days'],
                'event': df_targets.loc[cid, 'event'],
            })
        
        X_all = torch.stack(fused_list)
        y_df = pd.DataFrame(targets_filtered).set_index('case_id')
        
        # Cross-validated training
        seed_cis = []
        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            fold_cis = []
            
            for tr_idx, va_idx in skf.split(np.zeros(len(y_df)), y_df['event']):
                X_tr = X_all[tr_idx]
                X_va = X_all[va_idx]
                T_tr = torch.tensor(y_df.iloc[tr_idx]['survival_days'].values, dtype=torch.float32)
                E_tr = torch.tensor(y_df.iloc[tr_idx]['event'].values, dtype=torch.float32)
                T_va = torch.tensor(y_df.iloc[va_idx]['survival_days'].values, dtype=torch.float32)
                E_va = torch.tensor(y_df.iloc[va_idx]['event'].values, dtype=torch.float32)
                
                prognosis_name = self.config['phase_5_multimodal']['prognosis_proc']
                prognosis = get_prognosis_proc(
                    prognosis_name,
                    fused_dim=fusion.fused_dim,
                )
                result = prognosis.fit(
                    X_tr, T_tr, E_tr, X_va, T_va, E_va,
                    epochs=self.config['phase_5_multimodal'].get('prognosis_epochs', 200),
                    patience=self.config['phase_5_multimodal'].get('prognosis_patience', 20),
                    verbose=False,
                )
                fold_cis.append(result['best_val_cindex'])
            
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
        # 1. Encode cohort once (cached)
        self.encode_cohort(df_features, df_targets, modality_files)
        
        # 2. Default ablations: tabular alone + each pair + full
        if ablations is None:
            ablations = []
            if 'tabular' in self.modalities:
                ablations.append(['tabular'])
                if 'text' in self.modalities:
                    ablations.append(['tabular', 'text'])
                if 'vision' in self.modalities:
                    ablations.append(['tabular', 'vision'])
                if 'text' in self.modalities and 'vision' in self.modalities:
                    ablations.append(['tabular', 'text', 'vision'])
        
        results = []
        for subset in ablations:
            print(f"  Evaluating subset: {subset}")
            r = self.evaluate_combination(subset, df_targets, seeds, n_folds)
            r['subset_label'] = '+'.join(subset)
            results.append(r)
            if 'error' not in r:
                print(f"    C-index: {r['cindex_mean']:.4f} ± {r['cindex_std']:.4f} (n={r['n_cases']})")
            else:
                print(f"    ERROR: {r['error']}")
        
        return pd.DataFrame(results)
