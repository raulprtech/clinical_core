import time
from google import genai
from .base import BaseExplainer, ExplanationOutput
from .backends.oxigraph_db import OxigraphStore
from .backends.kuzu_db import KuzuStore

class GraphRAGExplainer(BaseExplainer):
    def __init__(self, config: dict):
        self.api_key = config.get('api_key')
        self.client = genai.Client(api_key=self.api_key)
        self.model = config.get('llm_model', 'gemini-2.5-flash')
        
        # Selección dinámica del backend desde el YAML
        backend_type = config.get('backend', 'oxigraph')
        if backend_type == 'oxigraph':
            from .backends.oxigraph_db import OxigraphStore
            self.graph = OxigraphStore()
        elif backend_type == 'kuzu':
            from .backends.kuzu_db import KuzuStore
            self.graph = KuzuStore()
        elif backend_type == 'rdflib':
            from .backends.rdflib_db import RDFLibStore
            self.graph = RDFLibStore()
        else:
            raise ValueError(f"Backend de grafo no soportado: {backend_type}")
        self.graph.load_knowledge()

    def explain(self, features, prediction) -> ExplanationOutput:
        start_time = time.time()
        
        # Extraer conceptos clínicos de las features multimodales
        concepts = features.get('vision_concepts', []) + features.get('text_concepts', [])
        kg_context = self.graph.query(concepts)
        
        # Prompt anclado a la ontología (G1 Trazabilidad)
        prompt = f"Explica el riesgo {prediction['risk']:.2f} usando estos nodos UMLS: {kg_context}"
        
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        
        return ExplanationOutput(
            text=response.text,
            anchor_nodes=[n['id'] for n in kg_context],
            confidence_flag="reliable" if prediction['confidence'] > 0.6 else "uncertain",
            latency_ms=(time.time() - start_time) * 1000
        )