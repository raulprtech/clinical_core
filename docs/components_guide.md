# Developer Guide: Adding Components

Follow these steps to add new capabilities to the CLINICAL-CORE ecosystem.

## 1. Implement the Component

Create a new file in the corresponding directory. For example, to add a new fusion strategy:
`code/components/processors/fusion/transformer_fusion.py`

```python
import torch

class FusionProc_Transformer:
    name = "fusion_transformer"
    
    def __init__(self, modalities, modality_dims=None, **kwargs):
        # Your init logic
        self.fused_dim = ...
        
    def fuse_one(self, modality_outputs):
        # modality_outputs: {name: (embedding, confidence)}
        # Return Tuple[torch.Tensor, float]
        return fused_vector, mean_confidence
```

## 2. Register Your Component

Edit `code/registry.py` to make your class visible to the configuration system.

```python
from components.processors.fusion.transformer_fusion import FusionProc_Transformer

FUSION_PROC_REGISTRY = {
    'fusion_baseline_concat': ...,
    'fusion_transformer': lambda modalities, **kw: FusionProc_Transformer(
        modalities=modalities, **kw
    ),
}
```

## 3. Update the Experiment Config

Activate your component in `code/configs/experiment_config.yaml`:

```yaml
phase_5_multimodal:
  fusion_proc: "fusion_transformer"
```

## Contracts & Requirements

- **Dimensionality**: Modality embeddings are expected to be 768-dimensional by default for consistent fusion.
- **Normalization**: Embeddings SHOULD be L2-normalized before being returned by a Connector.
- **Confidence**: Always return a confidence score between 0.0 (missing/failure) and 1.0 (perfect ingestion).
- **GPU/CPU**: Components should be agnostic or handle device placement internally (default is typically CPU for connectors).
