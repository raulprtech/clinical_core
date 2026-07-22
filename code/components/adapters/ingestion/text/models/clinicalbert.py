"""
TEXT-CONN: ClinicalBERT Connector
=================================

Baseline implementation: Docling (extraction) + ClinicalBERT (embedding).
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import hashlib
import warnings
import numpy as np
import torch

# ... (DoclingExtractor and ClinicalBertEmbedder from text_conn.py) ...
# I will copy the full content for completeness.

class DoclingExtractor:
    """Thin wrapper around Docling. Falls back to PyPDF2/raw text if unavailable."""
    def __init__(self):
        self._docling = None
        self._mode = self._init_backend()
    def _init_backend(self) -> str:
        try:
            from docling.document_converter import DocumentConverter
            self._docling = DocumentConverter(); return "docling"
        except ImportError: pass
        try: import PyPDF2; return "pypdf2"
        except ImportError: pass
        return "raw_only"
    def extract(self, source: Union[str, Path]) -> Tuple[str, float]:
        source = Path(source) if not isinstance(source, str) or Path(source).exists() else source
        if isinstance(source, str) and not Path(source).exists(): return source, 1.0
        source = Path(source)
        if not source.exists(): return "", 0.0
        suffix = source.suffix.lower()
        if suffix in ['.txt', '.md']:
            try: return source.read_text(encoding='utf-8', errors='ignore'), 1.0
            except Exception: return "", 0.0
        if suffix == '.pdf':
            if self._mode == "docling":
                try:
                    result = self._docling.convert(str(source))
                    return result.document.export_to_markdown(), 1.0
                except Exception as e: warnings.warn(f"Docling failed: {e}")
            if self._mode == "pypdf2":
                try:
                    import PyPDF2
                    with open(source, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = "\n".join(p.extract_text() or "" for p in reader.pages)
                    return text, 0.7
                except Exception: return "", 0.0
        return "", 0.0

class ClinicalBertEmbedder:
    """ClinicalBERT [CLS] token extractor with mock fallback."""
    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    EMBEDDING_DIM = 768
    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._mode = "uninitialized"  # "real", "mock", or "uninitialized"
    def _lazy_init(self) -> str:
        if self._mode != "uninitialized":
            return self._mode
        try:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModel.from_pretrained(self.MODEL_NAME)
            self._model.eval()
            self._mode = "real"
        except Exception as e:
            warnings.warn(f"Failed to load ClinicalBERT, falling back to mock: {e}")
            self._mode = "mock"
        return self._mode

    def embed(self, text: str) -> torch.Tensor:
        mode = self._lazy_init()
        if mode == "real":
            with torch.no_grad():
                inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                outputs = self._model(**inputs)
                return outputs.last_hidden_state[0, 0, :]
        else:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            return torch.tensor(rng.standard_normal(self.EMBEDDING_DIM).astype(np.float32))

class TextConn_Baseline:
    name = "text_baseline_docling_clinicalbert"
    output_dim = 768
    def __init__(self, **kwargs):
        self.extractor = DoclingExtractor()
        self.embedder = ClinicalBertEmbedder()
        self._min_text_length = kwargs.get('min_text_length', 50)
    def encode(self, source: Union[str, Path]) -> Tuple[torch.Tensor, float]:
        text, extraction_quality = self.extractor.extract(source)
        if not text or len(text) < self._min_text_length:
            return torch.zeros(self.output_dim), 0.0
        embedding = self.embedder.embed(text)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        length_factor = min(1.0, len(text) / 500.0)
        confidence = float(extraction_quality * length_factor)
        return embedding, confidence


def load_precomputed_text_embeddings(
    npz_path: Union[str, Path], output_dim: int = 768
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """Load the cache produced by ``tools/build_text_embeddings.py``."""
    path = Path(npz_path)
    with np.load(path, allow_pickle=True) as cache:
        required = {'case_ids', 'embeddings', 'confidence'}
        missing = required.difference(cache.files)
        if missing:
            raise ValueError(f"Text embedding cache is missing {sorted(missing)}: {path}")
        case_ids = cache['case_ids']
        embeddings = cache['embeddings']
        confidence = cache['confidence']
    if embeddings.ndim != 2 or embeddings.shape[1] != output_dim:
        raise ValueError(
            f"Text embeddings have shape {embeddings.shape}; expected [N, {output_dim}]"
        )
    if len(case_ids) != len(embeddings) or len(confidence) != len(embeddings):
        raise ValueError(f"Text embedding arrays are not aligned: {path}")

    result = {}
    for case_id, values, conf in zip(case_ids, embeddings, confidence):
        conf = float(conf)
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if conf <= 0.0 or not np.isfinite(conf):
            continue
        if not torch.isfinite(tensor).all() or float(tensor.norm()) == 0.0:
            continue
        result[str(case_id).strip().upper()] = (
            torch.nn.functional.normalize(tensor, p=2, dim=0),
            min(1.0, conf),
        )
    return result
