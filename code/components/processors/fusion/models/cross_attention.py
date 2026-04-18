import torch
import torch.nn as nn
from code.core.registry import register_component

@register_component("fusion_cross_attention")
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8):
        super().__init__()
        # La trampa matemática O(N^2) que asfixiará la RAM
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, vision_features, clinical_features):
        # Compara cada variable clínica contra cada característica visual
        attn_out, _ = self.cross_attn(query=clinical_features, 
                                      key=vision_features, 
                                      value=vision_features)
        return self.norm(clinical_features + attn_out)