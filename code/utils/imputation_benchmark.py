"""
TABULAR-CONN v2: Imputation Benchmark & Preprocessing
Phase 1 of Experimental Protocol — aligned with Protocol v8

Evaluates: Mean/Median, KNN, MICE
Metrics: K-S fidelity + C-index downstream with fixed Cox predictor
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy import stats
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# IMPUTATION STRATEGIES
# ============================================================

class ImputationStrategy:
    """Base class for imputation strategies."""
    name: str = "base"
    
    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
    
    def transform(self, X_val: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class MeanMedianImputer(ImputationStrategy):
    """Baseline: mean for continuous, median for skewed, mode for categorical."""
    name = "mean_median"
    
    def __init__(self):
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        self.numeric_cols = []
        self.categorical_cols = []
    
    def fit_transform(self, X: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> pd.DataFrame:
        self.numeric_cols = [c for c in numeric_cols if c in X.columns]
        self.categorical_cols = [c for c in categorical_cols if c in X.columns]
        
        result = X.copy()
        
        # Pre-fill columns that are 100% NaN — SimpleImputer drops them,
        # causing "Columns must be same length as key"
        for col in result.columns:
            if result[col].isna().all():
                result[col] = 0.0
        
        if self.numeric_cols:
            cols_with_data = [c for c in self.numeric_cols if not X[c].isna().all()]
            if cols_with_data:
                result[cols_with_data] = self.imputer_num.fit_transform(result[cols_with_data])
        if self.categorical_cols:
            cols_with_data = [c for c in self.categorical_cols if not X[c].isna().all()]
            if cols_with_data:
                result[cols_with_data] = self.imputer_cat.fit_transform(result[cols_with_data])
        return result
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        
        # Pre-fill columns that are 100% NaN
        for col in result.columns:
            if result[col].isna().all():
                result[col] = 0.0
        
        if self.numeric_cols:
            cols_with_data = [c for c in self.numeric_cols if c in result.columns and not X[c].isna().all()]
            if cols_with_data:
                result[cols_with_data] = self.imputer_num.transform(result[cols_with_data])
        if self.categorical_cols:
            cols_with_data = [c for c in self.categorical_cols if c in result.columns and not X[c].isna().all()]
            if cols_with_data:
                result[cols_with_data] = self.imputer_cat.transform(result[cols_with_data])
        return result


class KNNImputerStrategy(ImputationStrategy):
    """KNN Imputation — deterministic, FPGA-compatible."""
    name = "knn"
    
    def __init__(self, n_neighbors: int = 5):
        self.imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        self.columns = []
    
    def fit_transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.columns = X.columns.tolist()
        X_safe = X.copy()
        # Pre-fill columns that are 100% NaN to avoid imputer issues
        for col in X_safe.columns:
            if X_safe[col].isna().all():
                X_safe[col] = 0.0
        result = pd.DataFrame(
            self.imputer.fit_transform(X_safe),
            columns=self.columns, index=X.index
        )
        return result
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_safe = X.copy()
        for col in X_safe.columns:
            if X_safe[col].isna().all():
                X_safe[col] = 0.0
        result = pd.DataFrame(
            self.imputer.transform(X_safe),
            columns=self.columns, index=X.index
        )
        return result


class MICEImputerStrategy(ImputationStrategy):
    """MICE (Multiple Imputation by Chained Equations)."""
    name = "mice"
    
    def __init__(self, max_iter: int = 10, random_state: int = 42):
        self.imputer = IterativeImputer(
            max_iter=max_iter, random_state=random_state,
            sample_posterior=False  # Deterministic for reproducibility
        )
        self.columns = []
    
    def fit_transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.columns = X.columns.tolist()
        X_safe = X.copy()
        for col in X_safe.columns:
            if X_safe[col].isna().all():
                X_safe[col] = 0.0
        result = pd.DataFrame(
            self.imputer.fit_transform(X_safe),
            columns=self.columns, index=X.index
        )
        return result
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_safe = X.copy()
        for col in X_safe.columns:
            if X_safe[col].isna().all():
                X_safe[col] = 0.0
        result = pd.DataFrame(
            self.imputer.transform(X_safe),
            columns=self.columns, index=X.index
        )
        return result


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

class TabularPreprocessor:
    """
    Preprocesses TCGA-KIRC features:
    1. Identifies numeric vs categorical columns
    2. Creates missingness masks
    3. Applies chosen imputation
    4. Applies z-score normalization
    5. Computes confidence score
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_cols = []
        self.all_cols = []
        self.categorical_cols = []
        self.fitted = False
    
    def identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Separate numeric and categorical columns."""
        numeric = []
        categorical = []
        
        for col in df.columns:
            # Check unique non-null values
            unique = df[col].dropna().unique()
            if len(unique) <= 10:
                categorical.append(col)
            else:
                numeric.append(col)
        
        return numeric, categorical
    
    def create_missingness_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Binary mask: 1 = present, 0 = missing.
        This mask is appended to the feature vector.
        """
        mask = (~df.isna()).astype(float)
        mask.columns = [f"mask__{c}" for c in mask.columns]
        return mask
    
    def compute_confidence(self, mask: pd.DataFrame) -> pd.Series:
        """
        Confidence score: fraction of non-missing values.
        Weighted by clinical importance (staging > demographics > labs).
        """
        # Priority weights for clinical variables
        priority_vars = {
            'pathologic_stage': 2.0, 'pathologic_T': 2.0,
            'histologic_grade': 2.0, 'pathologic_M': 1.5,
            'age': 1.0, 'gender': 0.5,
            'hemoglobin': 1.5, 'serum_calcium': 1.5,
            'ldh': 1.5, 'karnofsky_score': 1.5,
        }
        
        weights = []
        for col in mask.columns:
            clean_col = col.replace('mask__', '')
            w = priority_vars.get(clean_col, 1.0)
            weights.append(w)
        
        weights = np.array(weights)
        weighted_mask = mask.values * weights[np.newaxis, :]
        confidence = weighted_mask.sum(axis=1) / weights.sum()
        
        return pd.Series(confidence, index=mask.index, name='confidence')
    
    def fit_transform(
        self, df_features: pd.DataFrame, 
        imputation_strategy: ImputationStrategy
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Full preprocessing pipeline.
        Returns: (processed_features, missingness_mask, confidence_scores)
        """
        self.all_cols = df_features.columns.tolist()
        self.numeric_cols, self.categorical_cols = self.identify_column_types(df_features)
        
        # Step 1: Create mask BEFORE imputation
        mask = self.create_missingness_mask(df_features)
        
        # Step 2: Compute confidence from mask
        confidence = self.compute_confidence(mask)
        
        # Step 3: Impute
        if isinstance(imputation_strategy, MeanMedianImputer):
            df_imputed = imputation_strategy.fit_transform(
                df_features, 
                numeric_cols=self.numeric_cols,
                categorical_cols=self.categorical_cols
            )
        else:
            df_imputed = imputation_strategy.fit_transform(df_features)
        
        # Step 4: Z-score normalize numeric columns
        if self.numeric_cols:
            cols_present = [c for c in self.numeric_cols if c in df_imputed.columns]
            df_imputed[cols_present] = self.scaler.fit_transform(df_imputed[cols_present])
        
        self.fitted = True
        self.imputer = imputation_strategy

        # Ensure output columns match input columns exactly
        # (guards against columns lost during imputation, e.g. race with 0% data)
        for col in self.all_cols:
            if col not in df_imputed.columns:
                df_imputed[col] = 0.0
        df_imputed = df_imputed[self.all_cols]

        return df_imputed, mask, confidence
    
    def transform(self, df_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Transform new data using fitted preprocessor."""
        assert self.fitted, "Must call fit_transform first"
        
        mask = self.create_missingness_mask(df_features)
        confidence = self.compute_confidence(mask)
        
        df_imputed = self.imputer.transform(df_features)
        
        if self.numeric_cols:
            cols_present = [c for c in self.numeric_cols if c in df_imputed.columns]
            df_imputed[cols_present] = self.scaler.transform(df_imputed[cols_present])
        
        # Ensure output columns match fitted columns exactly
        for col in self.all_cols:
            if col not in df_imputed.columns:
                df_imputed[col] = 0.0
        df_imputed = df_imputed[self.all_cols]

        return df_imputed, mask, confidence


# ============================================================
# IMPUTATION BENCHMARK (Phase 1)
# ============================================================

def run_imputation_benchmark(
    df_features: pd.DataFrame, 
    df_targets: pd.DataFrame,
    n_seeds: int = 5
) -> pd.DataFrame:
    """
    Phase 1 of experimental protocol.
    Evaluates imputation strategies with dual metric: K-S fidelity + C-index.
    
    Uses FIXED predictor (Cox-PH) to isolate imputation effect.
    """
    # Remove cases with missing survival data
    valid_mask = df_targets['survival_days'].notna() & (df_targets['survival_days'] > 0)
    df_feat = df_features.loc[valid_mask].copy()
    df_targ = df_targets.loc[valid_mask].copy()
    
    print(f"Imputation Benchmark: {len(df_feat)} cases with valid survival data")
    print(f"Events: {int(df_targ['event'].sum())}, Censored: {int(len(df_targ) - df_targ['event'].sum())}")
    
    strategies = {
        'mean_median': lambda: MeanMedianImputer(),
        'knn_5': lambda: KNNImputerStrategy(n_neighbors=5),
        'knn_10': lambda: KNNImputerStrategy(n_neighbors=10),
        'mice': lambda: MICEImputerStrategy(max_iter=10),
    }
    
    results = []
    
    for strategy_name, strategy_factory in strategies.items():
        print(f"\n--- Evaluating: {strategy_name} ---")
        
        seed_cindex = []
        seed_ks = []
        
        for seed in range(n_seeds):
            # Stratified split by event indicator
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            fold_cindex = []
            fold_ks_scores = []
            
            for train_idx, val_idx in skf.split(df_feat, df_targ['event']):
                X_train = df_feat.iloc[train_idx].copy()
                X_val = df_feat.iloc[val_idx].copy()
                y_train = df_targ.iloc[train_idx].copy()
                y_val = df_targ.iloc[val_idx].copy()
                
                # Store original distributions for K-S test
                original_distributions = {}
                for col in X_train.select_dtypes(include=[np.number]).columns:
                    valid_vals = X_train[col].dropna()
                    if len(valid_vals) > 10:
                        original_distributions[col] = valid_vals.values
                
                # Apply imputation
                strategy = strategy_factory()
                preprocessor = TabularPreprocessor()
                X_train_proc, _, _ = preprocessor.fit_transform(X_train, strategy)
                X_val_proc, _, _ = preprocessor.transform(X_val)
                
                # K-S test: compare imputed vs original distributions
                ks_scores = []
                for col, orig_vals in original_distributions.items():
                    if col in X_train_proc.columns:
                        imputed_vals = X_train_proc[col].values
                        stat, pval = stats.ks_2samp(orig_vals, imputed_vals)
                        ks_scores.append(stat)
                
                avg_ks = np.mean(ks_scores) if ks_scores else np.nan
                fold_ks_scores.append(avg_ks)
                
                # C-index with Cox-PH (FIXED predictor)
                try:
                    cox_df_train = X_train_proc.copy()
                    cox_df_train['T'] = y_train['survival_days'].values
                    cox_df_train['E'] = y_train['event'].values
                    
                    # Drop constant or all-NaN columns
                    valid_cols = [c for c in cox_df_train.columns 
                                  if c not in ['T', 'E'] and cox_df_train[c].std() > 1e-8]
                    cox_df_train = cox_df_train[valid_cols + ['T', 'E']]
                    
                    # Handle any remaining infinities
                    cox_df_train = cox_df_train.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    cph = CoxPHFitter(penalizer=0.1)
                    cph.fit(cox_df_train, duration_col='T', event_col='E')
                    
                    # Predict on validation
                    X_val_cox = X_val_proc[valid_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                    risk_scores = cph.predict_partial_hazard(X_val_cox).values.ravel()
                    
                    ci = concordance_index(
                        y_val['survival_days'].values,
                        -risk_scores,  # Negative because higher hazard = shorter survival
                        y_val['event'].values
                    )
                    fold_cindex.append(ci)
                except Exception as e:
                    print(f"    Cox failed on fold: {e}")
                    fold_cindex.append(np.nan)
            
            seed_cindex.append(np.nanmean(fold_cindex))
            seed_ks.append(np.nanmean(fold_ks_scores))
        
        mean_ci = np.nanmean(seed_cindex)
        std_ci = np.nanstd(seed_cindex)
        mean_ks = np.nanmean(seed_ks)
        std_ks = np.nanstd(seed_ks)
        
        print(f"  C-index: {mean_ci:.4f} ± {std_ci:.4f}")
        print(f"  K-S statistic (lower = better): {mean_ks:.4f} ± {std_ks:.4f}")
        
        results.append({
            'strategy': strategy_name,
            'cindex_mean': mean_ci,
            'cindex_std': std_ci,
            'ks_mean': mean_ks,
            'ks_std': std_ks,
        })
    
    results_df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("IMPUTATION BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    
    # Determine winner
    best_idx = results_df['cindex_mean'].idxmax()
    best = results_df.iloc[best_idx]
    print(f"\nRecommended strategy: {best['strategy']} (C-index: {best['cindex_mean']:.4f})")
    
    return results_df


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n = 100
    df_test = pd.DataFrame({
        'age': np.random.normal(60, 12, n),
        'grade': np.random.choice([1, 2, 3, 4], n),
        'stage': np.random.choice([1, 2, 3, 4], n),
    })
    # Introduce 20% missing
    for col in df_test.columns:
        mask = np.random.random(n) < 0.2
        df_test.loc[mask, col] = np.nan
    
    targets = pd.DataFrame({
        'survival_days': np.random.exponential(1000, n),
        'event': np.random.binomial(1, 0.3, n)
    })
    
    results = run_imputation_benchmark(df_test, targets, n_seeds=2)
