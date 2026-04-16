import rdflib
from rdflib.namespace import SKOS
from ..base import BaseGraphStore
from typing import List, Dict, Any

class RDFLibStore(BaseGraphStore):
    """
    Backend de Grafo Semántico usando RDFLib (Python Puro).
    Uso previsto: Baseline para benchmarks de latencia y entornos sin dependencias C/Rust.
    """
    def __init__(self):
        self.g = rdflib.Graph()
        self.UMLS = rdflib.Namespace("http://umls.org/")

    def load_knowledge(self, turtle_data: str = None) -> None:
        """Carga la ontología en memoria."""
        if turtle_data:
            self.g.parse(data=turtle_data, format="turtle")

    def query(self, concept_ids: List[str]) -> List[Dict[str, Any]]:
        """Recupera nodos relevantes usando SPARQL estándar W3C."""
        if not concept_ids:
            return []

        query_str = """
        PREFIX umls: <http://umls.org/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?id ?label ?def ?assocLabel
        WHERE {
            ?node rdfs:label ?label .
            OPTIONAL { ?node skos:definition ?def }
            OPTIONAL { 
                ?node umls:associated_with ?assoc . 
                ?assoc rdfs:label ?assocLabel 
            }
            BIND(REPLACE(STR(?node), "http://umls.org/", "") AS ?id)
            FILTER(?id IN (%s))
        }
        """ % ", ".join(f'"{id}"' for id in concept_ids)
        
        res = self.g.query(query_str)
        
        return [
            {
                'id': str(row.id), 
                'label': str(row.label), 
                'def': str(row.def) if row.def else "",
                'assoc': str(row.assocLabel) if row.assocLabel else ""
            } 
            for row in res
        ]