import os
import json
import logging
from typing import List, Dict, Set, Optional
import requests

logger = logging.getLogger(__name__)

class GDCDataFetcher:
    """
    Programmatic utility for interacting with the GDC API to search and download datasets.
    """
    
    API_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
    API_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"

    def __init__(self, project_id: str = "TCGA-KIRC"):
        self.project_id = project_id
        
    def search_files(
        self,
        data_types: List[str],
        data_formats: Optional[List[str]] = None,
        limit: int = 10000,
        target_case_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Searches the GDC API for files matching the specified data types for the target project.
        
        Args:
            data_types: List of GDC data types (e.g. ['Slide Image', 'Clinical Supplement'])
            data_formats: List of specific data formats to restrict (e.g. ['BCR XML', 'SVS', 'TSV'])
            limit: Maximum number of files to return from the API endpoint.
            target_case_ids: Optional list of case_ids (cohort intersection constraint). If provided,
                             the API will only return results for these exact patients.

        Returns:
            A list of dictionary objects representing metadata for matching files.
        """
        if not data_types:
            logger.warning("No data_types explicitly provided for search.")
            return []

        logger.info(f"Searching GDC files for project {self.project_id} and types: {data_types}")
        
        content_filters = [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [self.project_id]}},
            {"op": "in", "content": {"field": "data_type", "value": data_types}}
        ]

        if data_formats:
            content_filters.append({"op": "in", "content": {"field": "data_format", "value": data_formats}})
            
        if target_case_ids:
            content_filters.append({"op": "in", "content": {"field": "cases.case_id", "value": target_case_ids}})

        filtros = {
            "op": "and",
            "content": content_filters
        }

        parametros = {
            "filters": json.dumps(filtros),
            "fields": "file_id,cases.case_id,data_type,data_format,file_name",
            "format": "JSON",
            "size": str(limit)
        }

        try:
            response = requests.get(self.API_FILES_ENDPOINT, params=parametros)
            response.raise_for_status()
            
            data = response.json()
            hits = data.get("data", {}).get("hits", [])
            
            # We must group by case_id and filter cases that possess ALL requested data_types
            grouped = self._filter_cases_by_intersection(hits, set(data_types))
            
            logger.info(f"Found {len(grouped)} valid files crossing required constraints.")
            return grouped
            
        except Exception as e:
            logger.error(f"Failed to query GDC API: {e}")
            raise

    def _filter_cases_by_intersection(self, hits: List[Dict], target_types: Set[str]) -> List[Dict]:
        """
        Ensures a patient (case_id) contains all specified data_types, dropping incomplete sets.
        """
        pacientes = {}
        for archivo in hits:
            cases = archivo.get("cases", [{}])
            if not cases:
                continue
            case_id = cases[0].get("case_id")
            if not case_id:
                continue

            if case_id not in pacientes:
                pacientes[case_id] = {"tipos_disponibles": set(), "archivos": []}

            dt = archivo.get("data_type")
            pacientes[case_id]["tipos_disponibles"].add(dt)
            if dt in target_types:
                pacientes[case_id]["archivos"].append(archivo)

        archivos_finales = []
        for case_id, datos in pacientes.items():
            if target_types.issubset(datos["tipos_disponibles"]):
                archivos_finales.extend(datos["archivos"])
                
        return archivos_finales

    def download_files(self, files_metadata: List[Dict], base_output_dir: str) -> None:
        """
        Downloads a list of file metadata objects to their canonical category folders.

        Args:
            files_metadata: List of file dictionaries obtained from `search_files`.
            base_output_dir: Root directory to save the GDC data.
        """
        if not files_metadata:
            logger.info("No files to download.")
            return

        logger.info(f"Starting download of {len(files_metadata)} files to {base_output_dir}")

        for i, archivo in enumerate(files_metadata, 1):
            file_id = archivo.get('file_id')
            file_name = archivo.get('file_name', f"file_{file_id}")
            data_type = archivo.get("data_type", "unknown").replace(" ", "").lower()
            
            ruta_carpeta = os.path.join(base_output_dir, data_type)
            os.makedirs(ruta_carpeta, exist_ok=True)
            
            ruta_archivo = os.path.join(ruta_carpeta, file_name)

            if os.path.exists(ruta_archivo):
                logger.debug(f"Skipping {file_name}, already exists. ({i}/{len(files_metadata)})")
                continue
                
            try:
                res_descarga = requests.get(f"{self.API_DATA_ENDPOINT}/{file_id}", stream=True)
                res_descarga.raise_for_status()
                
                with open(ruta_archivo, "wb") as f:
                    for bloque in res_descarga.iter_content(chunk_size=8192):
                        if bloque:
                            f.write(bloque)
                logger.info(f"Downloaded {file_name} ({i}/{len(files_metadata)})")
                
            except Exception as e:
                logger.error(f"Error downloading file {file_id}: {e}")
                
        logger.info("Download process completed.")

    def save_cohort_manifest(self, files_metadata: List[Dict], output_path: str) -> None:
        """
        Extracts all unique case_ids from the file hits and dumps them to a JSON manifest.
        This manifest can be used in subsequent runs via the `target_case_ids` parameter
        to force the download of new modalities strictly mapped to this cohort.
        """
        case_ids = set()
        for archivo in files_metadata:
            cases = archivo.get("cases", [])
            for c in cases:
                if "case_id" in c:
                    case_ids.add(c["case_id"])
                    
        manifest = {
            "project_id": self.project_id,
            "case_ids": list(case_ids),
            "generated_from_files": len(files_metadata)
        }
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=4)
        logger.info(f"Cohort manifest saved with {len(case_ids)} unique case_ids at {output_path}")

    @staticmethod
    def load_cohort_manifest(manifest_path: str) -> List[str]:
        """ Helper utility to read a case_ids list from a standard JSON manifest. """
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
            
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        return manifest.get("case_ids", [])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Dry run testing of GDCDataFetcher")
    fetcher = GDCDataFetcher()
    # Test only 2 case cross, forcing BCR XML format.
    files = fetcher.search_files(["Clinical Supplement"], data_formats=["BCR XML"], limit=2)
    fetcher.save_cohort_manifest(files, "dry_run_manifest.json")
    print(f"Sample hits: {files}")
