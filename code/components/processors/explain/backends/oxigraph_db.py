import pyoxigraph
from ..base import BaseGraphStore

class OxigraphStore(BaseGraphStore):
    def __init__(self):
        self.store = pyoxigraph.Store()

    def load_knowledge(self, turtle_data: str = None):
        # Aquí cargaríamos el TTL de UMLS especializado
        if turtle_data:
            self.store.load(turtle_data.encode('utf-8'), "text/turtle")

    def query(self, concept_ids):
        query_str = """
        PREFIX umls: <http://umls.org/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?id ?label ?def ?assocLabel
        WHERE {
            ?node rdfs:label ?label .
            OPTIONAL { ?node skos:definition ?def }
            OPTIONAL { ?node umls:associated_with ?assoc . ?assoc rdfs:label ?assocLabel }
            BIND(REPLACE(STR(?node), "http://umls.org/", "") AS ?id)
            FILTER(?id IN (%s))
        }
        """ % ", ".join(f'"{id}"' for id in concept_ids)
        
        return [{'id': s['id'].value, 'label': s['label'].value, 
                 'def': s.get('def').value if 'def' in s else "",
                 'assoc': s.get('assocLabel').value if 'assocLabel' in s else ""} 
                for s in self.store.query(query_str)]