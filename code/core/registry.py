

from components.adapters.ingestion.tabular.utils.imputation_benchmark import (
    MeanMedianImputer,
    KNNImputerStrategy,
    MICEImputerStrategy,
)
from components.adapters.ingestion.tabular.models.cox_baseline import VariantA_CoxBaseline
from components.adapters.ingestion.tabular.models.linear_compact import VariantC_LinearEncoder
from components.adapters.ingestion.tabular.models.ft_transformer import build_ft_transformer

from components.adapters.ingestion.text.models.clinicalbert import TextConn_Baseline
from components.adapters.ingestion.vision.models.stunet import VisionConn_Baseline
from components.procesors.fusion.models.concatenation import FusionProc_Concatenation
from components.procesors.prognosis.models.linear_cox import PrognosisProc_LinearCox

# from components.processors.explain.graph_rag_explainer import GraphRAGExplainer
# ============================================================
# IMPUTATION STRATEGIES
# ============================================================
IMPUTATION_REGISTRY = {
    'mean_median': lambda: MeanMedianImputer(),
    'knn_5': lambda: KNNImputerStrategy(n_neighbors=5),
    'knn_10': lambda: KNNImputerStrategy(n_neighbors=10),
    'mice': lambda: MICEImputerStrategy(max_iter=10),
}


# ============================================================
# TABULAR-CONN ENCODER VARIANTS (Phase 2)
# ============================================================
VARIANT_REGISTRY = {
    'cox_baseline': lambda input_dim, output_dim, **kw: VariantA_CoxBaseline(
        input_dim=input_dim, output_dim=output_dim
    ),
    'linear_compact': lambda input_dim, output_dim, **kw: VariantC_LinearEncoder(
        input_dim=input_dim,
        hidden_dim=kw.get('hidden_dim', 128),
        output_dim=output_dim
    ),
    'ft_transformer': lambda input_dim, output_dim, **kw: build_ft_transformer(
        input_dim=input_dim,
        output_dim=output_dim,
        **kw
    ),
}


# ============================================================
# TEXT-CONN IMPLEMENTATIONS
# ============================================================
TEXT_CONN_REGISTRY = {
    'text_baseline_docling_clinicalbert': lambda **kw: TextConn_Baseline(**kw),
}


# ============================================================
# VISION-CONN IMPLEMENTATIONS
# ============================================================
VISION_CONN_REGISTRY = {
    'vision_baseline_stunet_radiomics': lambda **kw: VisionConn_Baseline(**kw),
}


# ============================================================
# FUSION-PROC IMPLEMENTATIONS
# ============================================================
FUSION_PROC_REGISTRY = {
    'fusion_baseline_concat': lambda modalities, modality_dims=None, **kw: FusionProc_Concatenation(
        modalities=modalities, modality_dims=modality_dims, **kw
    ),
}


# ============================================================
# PROGNOSIS-PROC IMPLEMENTATIONS
# ============================================================
PROGNOSIS_PROC_REGISTRY = {
    'prognosis_baseline_linear_cox': lambda fused_dim, **kw: PrognosisProc_LinearCox(
        fused_dim=fused_dim, **kw
    ),
}


# # ============================================================
# # EXPLAIN-PROC IMPLEMENTATIONS
# # ============================================================
# EXPLAIN_PROC_REGISTRY = {
#     'explain_graph_rag': lambda **kw: GraphRAGExplainer(**kw),
# }


# ============================================================
# UNIFIED LOOKUP API
# ============================================================

def _lookup(registry: dict, name: str, category: str):
    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(f"{category} '{name}' not found. Available: {available}")
    return registry[name]


def get_imputation(name: str):
    return _lookup(IMPUTATION_REGISTRY, name, "Imputation")()


def get_variant(name: str, input_dim: int, output_dim: int, **kwargs):
    return _lookup(VARIANT_REGISTRY, name, "Variant")(input_dim, output_dim, **kwargs)


def get_text_conn(name: str, **kwargs):
    return _lookup(TEXT_CONN_REGISTRY, name, "TextConn")(**kwargs)


def get_vision_conn(name: str, **kwargs):
    return _lookup(VISION_CONN_REGISTRY, name, "VisionConn")(**kwargs)


def get_fusion_proc(name: str, modalities, modality_dims=None, **kwargs):
    return _lookup(FUSION_PROC_REGISTRY, name, "FusionProc")(
        modalities=modalities, modality_dims=modality_dims, **kwargs
    )


def get_prognosis_proc(name: str, fused_dim: int, **kwargs):
    return _lookup(PROGNOSIS_PROC_REGISTRY, name, "PrognosisProc")(
        fused_dim=fused_dim, **kwargs
    )


def list_components() -> dict:
    return {
        'imputation': sorted(IMPUTATION_REGISTRY.keys()),
        'variants': sorted(VARIANT_REGISTRY.keys()),
        'text_conn': sorted(TEXT_CONN_REGISTRY.keys()),
        'vision_conn': sorted(VISION_CONN_REGISTRY.keys()),
        'fusion_proc': sorted(FUSION_PROC_REGISTRY.keys()),
        'prognosis_proc': sorted(PROGNOSIS_PROC_REGISTRY.keys()),
    }
