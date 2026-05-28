"""
Download the Kefeli TCGA-Reports corpus and filter to TCGA-KIRC.

Source: GitHub tatonetti-lab/tcga-path-reports (CC BY 4.0).
Citation: Kefeli & Tatonetti 2024, "TCGA-Reports: A machine-readable pathology
report resource for benchmarking text-based AI models." Cell Patterns.

Steps:
  1. Download TCGA_Reports.csv.zip if not already cached.
  2. Unzip into data/raw/text_reports_kefeli/.
  3. Auto-detect the report-text column and the case_id column (the published
     schema names have varied between releases).
  4. Report counts per TCGA project; persist a sidecar JSON with download URL,
     date, license, and per-project counts so Paper 2 can cite correctly.
  5. Filter to TCGA-KIRC case_ids and write a separate file
     `TCGA_Reports_KIRC.csv` for fast loading downstream.

Usage:
  python3 tools/download_kefeli_corpus.py
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("download_kefeli")

DEFAULT_URL = "https://github.com/tatonetti-lab/tcga-path-reports/raw/main/TCGA_Reports.csv.zip"
DEFAULT_OUT_DIR = REPO_ROOT.parent / "data" / "raw" / "text_reports_kefeli"

LICENSE = "CC BY 4.0 (Mendeley DOI 10.17632/hyg5xkznpx.1)"
CITATION = (
    "Kefeli J, Tatonetti N. TCGA-Reports: A machine-readable pathology report "
    "resource for benchmarking text-based AI models. Cell Patterns 2024."
)


def detect_columns(df: pd.DataFrame) -> dict:
    """Auto-detect the case_id and report_text columns by content."""
    cols = list(df.columns)
    case_col = None
    text_col = None

    # Case ID column: look for TCGA-XX-XXXX shape on > 80% of rows.
    for c in cols:
        try:
            ser = df[c].astype(str)
            hits = ser.str.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}").mean()
        except Exception:
            continue
        if hits > 0.80:
            case_col = c
            break

    # Text column: longest median string content among remaining columns.
    candidates = [c for c in cols if c != case_col]
    best_med = 0
    for c in candidates:
        try:
            ser = df[c].astype(str)
            med_len = ser.str.len().median()
        except Exception:
            continue
        if med_len and med_len > best_med:
            best_med = med_len
            text_col = c

    return {"case_id_col": case_col, "report_text_col": text_col,
            "median_text_chars": float(best_med)}


def extract_project_id(case_id: str) -> str | None:
    """TCGA-XX-XXXX -> TCGA-XX (which maps to the project, e.g. TCGA-XX is KIRC code)."""
    if not isinstance(case_id, str):
        return None
    parts = case_id.split("-")
    if len(parts) >= 2 and parts[0] == "TCGA":
        return f"TCGA-{parts[1]}"
    return None


# TCGA tissue source site code → cancer project (selected — KIRC is what we need).
KIRC_TSS_CODES = {
    # See https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes
    # Subset that map to KIRC (kidney renal clear cell carcinoma):
    "3Z", "6D", "A3", "AK", "AS", "B0", "B2", "B4", "B8", "BP",
    "CB", "CJ", "CW", "CZ", "DV", "EU", "G6", "GK", "MM", "MW",
    "T7",
}


def is_kirc_case(case_id: str) -> bool:
    if not isinstance(case_id, str):
        return False
    parts = case_id.split("-")
    if len(parts) >= 2 and parts[0] == "TCGA":
        return parts[1] in KIRC_TSS_CODES
    return False


def download_zip(url: str, dest: Path) -> Path:
    if dest.exists():
        logger.info(f"ZIP already present: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    logger.info(f"Downloading {url} ...")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
    logger.info(f"Saved {dest.stat().st_size / 1e6:.1f} MB to {dest}")
    return dest


def unzip(zip_path: Path, out_dir: Path) -> Path:
    """Extract; return the path to the main CSV inside."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        logger.info(f"ZIP contents: {names}")
        zf.extractall(out_dir)
    # Find a .csv file at the top level
    csvs = [p for p in out_dir.rglob("*.csv") if p.is_file()]
    if not csvs:
        raise RuntimeError(f"No .csv files found inside {zip_path}")
    if len(csvs) > 1:
        # Prefer one whose name contains "TCGA"
        preferred = [p for p in csvs if "TCGA" in p.name]
        csv_path = preferred[0] if preferred else csvs[0]
        logger.info(f"Multiple CSVs found; using {csv_path}")
    else:
        csv_path = csvs[0]
    return csv_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if the ZIP is already present.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.out_dir / "TCGA_Reports.csv.zip"

    if args.force and zip_path.exists():
        zip_path.unlink()

    download_zip(args.url, zip_path)
    csv_path = unzip(zip_path, args.out_dir)
    logger.info(f"Extracted CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"CSV loaded: shape={df.shape}, columns={list(df.columns)}")

    detected = detect_columns(df)
    case_col = detected["case_id_col"]
    text_col = detected["report_text_col"]
    if not case_col or not text_col:
        raise RuntimeError(
            f"Could not auto-detect case_id and text columns. "
            f"Found: {detected}"
        )
    logger.info(f"Detected columns: case_id='{case_col}', report_text='{text_col}'")
    logger.info(f"Median report length: {detected['median_text_chars']:.0f} chars")

    df = df.rename(columns={case_col: "case_id", text_col: "report_text"})
    # Some Kefeli releases use the patient_id, others the slide-level barcode.
    # We canonicalize to TCGA-XX-XXXX (patient barcode), the first 12 chars.
    df["case_id_full"] = df["case_id"]
    df["case_id"] = df["case_id"].astype(str).str.slice(0, 12)
    df["project_id"] = df["case_id"].apply(extract_project_id)

    # Per-project counts
    proj_counts = df["project_id"].value_counts().to_dict()
    logger.info("Per-project counts (top 10):")
    for proj, cnt in sorted(proj_counts.items(), key=lambda kv: -kv[1])[:10]:
        logger.info(f"  {proj}: {cnt}")

    # KIRC subset
    df_kirc = df[df["case_id"].apply(is_kirc_case)].copy()
    n_kirc_reports = len(df_kirc)
    n_kirc_cases = df_kirc["case_id"].nunique()
    logger.info(f"TCGA-KIRC: {n_kirc_reports} reports across {n_kirc_cases} unique cases")

    out_csv = args.out_dir / "TCGA_Reports_KIRC.csv"
    df_kirc[["case_id", "case_id_full", "report_text"]].to_csv(out_csv, index=False)
    logger.info(f"KIRC subset written: {out_csv}")

    sidecar = {
        "download_url": args.url,
        "download_date_utc": datetime.utcnow().isoformat() + "Z",
        "license": LICENSE,
        "citation": CITATION,
        "zip_md5_hint": "see Zenodo record 10.5281/zenodo.10452345 for code MD5; data MD5 not published",
        "raw_csv": str(csv_path),
        "detected_columns": detected,
        "n_reports_total": int(len(df)),
        "n_unique_cases_total": int(df["case_id"].nunique()),
        "per_project_report_counts": {k: int(v) for k, v in proj_counts.items() if k},
        "kirc_subset": {
            "n_reports": int(n_kirc_reports),
            "n_unique_cases": int(n_kirc_cases),
            "csv_path": str(out_csv),
        },
    }
    sidecar_path = args.out_dir / "TCGA_Reports_KIRC.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    logger.info(f"Sidecar: {sidecar_path}")

    print()
    print("=" * 60)
    print("KEFELI TCGA-REPORTS DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Total reports:        {len(df)}")
    print(f"Total unique cases:   {df['case_id'].nunique()}")
    print(f"TCGA-KIRC reports:    {n_kirc_reports}")
    print(f"TCGA-KIRC cases:      {n_kirc_cases}")
    print(f"License:              {LICENSE}")
    print(f"KIRC subset CSV:      {out_csv}")
    print(f"Sidecar JSON:         {sidecar_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
