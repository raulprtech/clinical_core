import torch
import torch.nn as nn
from code.core.registry import register_component

@register_component("fusion_optimal_transport")
class OptimalTransportFusion(nn.Module):
    def __init__(self, embed_dim=128, sinkhorn_iters=5):
        super().__init__()
        self.iters = sinkhorn_iters
        self.epsilon = 0.1
        self.proj_vision = nn.Linear(embed_dim, embed_dim)
        self.proj_clinic = nn.Linear(embed_dim, embed_dim)

    def forward(self, vision_features, clinical_features):
        v = self.proj_vision(vision_features)
        c = self.proj_clinic(clinical_features)
        
        # 1. Matriz de costo: Qué tan lejos está la imagen del texto
        cost_matrix = torch.cdist(v, c, p=2)
        
        # 2. Transporte Óptimo (Ruta de menor esfuerzo)
        K = torch.exp(-cost_matrix / self.epsilon)
        
        # Fusión alineada por el plan de transporte óptimo
        aligned_fusion = v + torch.bmm(K, c) 
        
        # [!] Aquí entra TurboLatent: Rotación de Hadamard antes de cuantizar
        return aligned_fusion