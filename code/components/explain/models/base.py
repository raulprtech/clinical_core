from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ExplanationOutput:
    text: str
    anchor_nodes: List[str]
    confidence_flag: str  # 'reliable', 'uncertain', 'rejected'
    latency_ms: float

class BaseGraphStore(ABC):
    @abstractmethod
    def load_knowledge(self, data_path: str = None) -> None:
        pass
    
    @abstractmethod
    def query(self, concept_ids: List[str]) -> List[Dict[str, Any]]:
        pass

class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, features: Any, prediction: Any) -> ExplanationOutput:
        pass