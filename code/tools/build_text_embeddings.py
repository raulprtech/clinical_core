"""
Pre-compute Bio_ClinicalBERT [CLS] embeddings for the project's TCGA-KIRC
cohort using the Kefeli TCGA-Reports corpus.

Outputs a single `.npz` aligned to the project's cohort (NOT to the Kefeli
corpus order). Each row corresponds to a case_id from --case-ids-from; if the
case has no Kefeli report, the row is zeros with confidence 0.

Steps:
  1. Load case_ids from --case-ids-from (raw_targets.csv from a previous run).
  2. Load Kefeli KIRC CSV from --kefeli-csv.
  3. Materialize each cohort case's report text to a temp file and call
     TextConn_Baseline().encode(temp_path) (real Bio_ClinicalBERT).
  4. Stack into shape [N, 768], track confidence [N].
  5. Assert that the loaded ClinicalBERT is REAL (not the mock fallback).
  6. Persist to data/embeddings/text_embeddings_TCGA-KIRC_<YYYYMMDD>.npz
     with a JSON sidecar describing the build.

Usage:
  python3 tools/build_text_embeddings.py \\
    --case-ids-from /path/to/raw_targets.csv \\
    --kefeli-csv data/raw/text_reports_kefeli/TCGA_Reports_KIRC.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from components.adapters.ingestion.text.models.clinicalbert import (  # noqa: E402
    TextConn_Baseline,
)

logger = logging.getLogger("build_text_embeddings")

DEFAULT_OUT_DIR = REPO_ROOT.parent / "data" / "embeddings"


def load_cohort_case_ids(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "case_id" in df.columns:
        ids = df["case_id"].astype(str).str.strip().tolist()
    elif df.index.name == "case_id":
        ids = df.index.astype(str).str.strip().tolist()
    else:
        raise ValueError(f"Could not find 'case_id' column in {path}")
    # Keep TCGA-XX-XXXX shape only
    ids = [i for i in ids if len(i) == 12 and i.startswith("TCGA-")]
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-ids-from", required=True, type=Path)
    ap.add_argument("--kefeli-csv", required=True, type=Path)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    ap.add_argument("--project-id", default="TCGA-KIRC")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.utcnow().strftime("%Y%m%d")
    npz_path = args.out_dir / f"text_embeddings_{args.project_id}_{today}.npz"
    sidecar_path = npz_path.with_suffix(".json")

    if npz_path.exists() and not args.force:
        logger.info(f"Embedding cache already exists: {npz_path}. Use --force to rebuild.")
        data = np.load(npz_path)
        print(f"\nExisting cache: {npz_path}  shape={data['embeddings'].shape}")
        print(f"  cases with text (conf > 0): {(data['confidence'] > 0).sum()}")
        return 0

    cohort_case_ids = load_cohort_case_ids(args.case_ids_from)
    logger.info(f"Loaded {len(cohort_case_ids)} cohort case_ids from {args.case_ids_from}")

    kefeli = pd.read_csv(args.kefeli_csv)
    if "case_id" not in kefeli.columns or "report_text" not in kefeli.columns:
        raise ValueError(
            f"Kefeli CSV missing expected columns. Have: {list(kefeli.columns)}"
        )
    kefeli_lookup = dict(zip(kefeli["case_id"], kefeli["report_text"]))
    logger.info(f"Loaded {len(kefeli_lookup)} reports from {args.kefeli_csv}")

    n_intersect = sum(1 for c in cohort_case_ids if c in kefeli_lookup)
    logger.info(f"Cohort ∩ Kefeli: {n_intersect} / {len(cohort_case_ids)} cases have a report")

    # Instantiate the encoder once, force the model to load on the first encode.
    logger.info("Initializing TextConn_Baseline (will trigger Bio_ClinicalBERT download "
                "on first invocation, ~440 MB)...")
    encoder = TextConn_Baseline()
    # NOTE: the TextConn_Baseline wrapper does NOT move inputs to GPU inside
    # its embed() method, so we keep the model on CPU to avoid a device
    # mismatch. For 525-ish reports at <= 512 tokens, CPU inference is fine
    # (a few minutes total). Production V2 might rewrite the wrapper to be
    # device-aware, but V1 prioritizes simplicity.
    encoder.embedder._lazy_init()
    if encoder.embedder._mode == "real":
        encoder.embedder._model.eval()
        logger.info("Bio_ClinicalBERT loaded on CPU (intentional — see comment in tools).")

    # First-call sanity: encode one report and verify we are NOT in mock mode.
    first_present = next((c for c in cohort_case_ids if c in kefeli_lookup), None)
    if first_present is None:
        raise RuntimeError("No cohort case has a Kefeli report — refusing to build a "
                           "fully-empty cache.")
    logger.info(f"Probe encode on {first_present} to verify real ClinicalBERT...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        tf.write(str(kefeli_lookup[first_present]))
        probe_path = Path(tf.name)
    probe_emb, probe_conf = encoder.encode(probe_path)
    probe_path.unlink(missing_ok=True)
    mode = encoder.embedder._mode
    logger.info(f"Probe: mode={mode}, emb_norm={probe_emb.norm():.4f}, conf={probe_conf:.4f}")
    if mode != "real":
        raise RuntimeError(
            f"Bio_ClinicalBERT failed to load (fallback mode='{mode}'). "
            f"Refusing to build a cache from mock embeddings. Fix the HuggingFace "
            f"download/network issue and retry."
        )
    assert torch.allclose(probe_emb.norm(), torch.tensor(1.0), atol=1e-2), (
        f"Embedding not L2-normalized: norm={probe_emb.norm()}"
    )

    # Loop over the cohort
    N = len(cohort_case_ids)
    embeddings = np.zeros((N, 768), dtype=np.float32)
    confidence = np.zeros(N, dtype=np.float32)
    n_done = 0
    n_missing = 0

    for i, cid in enumerate(cohort_case_ids):
        report = kefeli_lookup.get(cid)
        if report is None or not isinstance(report, str) or len(report) < 50:
            n_missing += 1
            continue
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
            tf.write(report)
            tmp_path = Path(tf.name)
        try:
            emb, conf = encoder.encode(tmp_path)
            embeddings[i] = emb.cpu().numpy().astype(np.float32)
            confidence[i] = float(conf)
            n_done += 1
        finally:
            tmp_path.unlink(missing_ok=True)
        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{N} encoded (done={n_done}, missing={n_missing})")

    logger.info(f"Done. Encoded: {n_done}, missing: {n_missing}, total: {N}")

    np.savez(
        npz_path,
        embeddings=embeddings,
        confidence=confidence,
        case_ids=np.array(cohort_case_ids, dtype=object),
    )
    logger.info(f"Saved: {npz_path}")

    # Stats
    mean_conf = float(confidence[confidence > 0].mean()) if (confidence > 0).any() else 0.0
    emb_norms_real = np.linalg.norm(embeddings[confidence > 0], axis=1)
    sidecar = {
        "project_id": args.project_id,
        "build_date_utc": datetime.utcnow().isoformat() + "Z",
        "encoder": "emilyalsentzer/Bio_ClinicalBERT (off-the-shelf, no MLM, no fine-tune)",
        "encoder_truncation_tokens": 512,
        "embedding_dim": 768,
        "embedding_l2_normalized": True,
        "case_ids_source": str(args.case_ids_from),
        "kefeli_source": str(args.kefeli_csv),
        "kefeli_license": "CC BY 4.0 (Mendeley DOI 10.17632/hyg5xkznpx.1)",
        "kefeli_citation": (
            "Kefeli J, Tatonetti N. TCGA-Reports: A machine-readable pathology "
            "report resource for benchmarking text-based AI models. "
            "Cell Patterns 2024."
        ),
        "n_cohort": int(N),
        "n_with_report": int(n_done),
        "n_missing_report": int(n_missing),
        "mean_confidence_when_present": float(mean_conf),
        "embedding_norm_mean": float(emb_norms_real.mean()) if len(emb_norms_real) else 0.0,
        "embedding_norm_std": float(emb_norms_real.std()) if len(emb_norms_real) else 0.0,
        "npz_path": str(npz_path),
        "caveats": [
            "Truncation at 512 tokens may drop content from late portions of "
            "long pathology reports (mean ~2854 chars in the Kefeli release).",
            "Single global cache: re-build when cohort changes or Bio_ClinicalBERT "
            "version changes.",
            "V1: off-the-shelf only — no MLM in-cohort, no Cox fine-tune.",
        ],
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    logger.info(f"Sidecar: {sidecar_path}")

    print()
    print("=" * 60)
    print("TEXT EMBEDDING CACHE COMPLETE")
    print("=" * 60)
    print(f"Cohort size:                {N}")
    print(f"Cases with report:          {n_done}")
    print(f"Cases missing report:       {n_missing}")
    print(f"Mean confidence (present):  {mean_conf:.4f}")
    print(f"NPZ:                        {npz_path}")
    print(f"Sidecar:                    {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
