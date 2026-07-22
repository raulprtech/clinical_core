"""
Component registries for the leakage-survival-protocol release.

Trimmed from the parent project: only the tabular pieces relevant to the
paper are exposed. Multimodal / fusion / text / vision / prognosis
registries are intentionally excluded.
"""
from src.preprocessing.imputation import (
    MeanMedianImputer,
    KNNImputerStrategy,
    MICEImputerStrategy,
)
from src.models.cox_baseline import VariantA_CoxBaseline
from src.models.linear_compact import VariantC_LinearEncoder
from src.models.ft_transformer import build_ft_transformer


# ============================================================
# IMPUTATION STRATEGIES
# ============================================================
IMPUTATION_REGISTRY = {
    'mean_median': lambda: MeanMedianImputer(),
    'knn_5':       lambda: KNNImputerStrategy(n_neighbors=5),
    'knn_10':      lambda: KNNImputerStrategy(n_neighbors=10),
    'mice':        lambda: MICEImputerStrategy(max_iter=10),
}


# ============================================================
# TABULAR-CONN ENCODER VARIANTS
# ============================================================
VARIANT_REGISTRY = {
    'cox_baseline':   lambda input_dim, output_dim, **kw: VariantA_CoxBaseline(
        input_dim=input_dim, output_dim=output_dim,
    ),
    'linear_compact': lambda input_dim, output_dim, **kw: VariantC_LinearEncoder(
        input_dim=input_dim,
        hidden_dim=kw.get('hidden_dim', 128),
        output_dim=output_dim,
    ),
    'ft_transformer': lambda input_dim, output_dim, **kw: build_ft_transformer(
        input_dim=input_dim, output_dim=output_dim, **kw
    ),
}


def _lookup(registry: dict, name: str, category: str):
    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(f"{category} '{name}' not found. Available: {available}")
    return registry[name]


def get_imputation(name: str):
    return _lookup(IMPUTATION_REGISTRY, name, "Imputation")()


def get_variant(name: str, input_dim: int, output_dim: int, **kwargs):
    return _lookup(VARIANT_REGISTRY, name, "Variant")(input_dim, output_dim, **kwargs)


def list_components() -> dict:
    return {
        'imputation': sorted(IMPUTATION_REGISTRY.keys()),
        'variants':   sorted(VARIANT_REGISTRY.keys()),
    }
