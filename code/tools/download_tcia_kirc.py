"""Select and download one TCIA CT/MR series per TCGA-KIRC patient.

The ranking reproduces the VISION-L0 Colab notebooks: prefer CT, then an
abdomen/renal description, then the largest slice count, while excluding
localizers and other non-diagnostic series. Downloads are resumable through a
completion marker stored beside every extracted DICOM series.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
import requests


BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "TCGA-KIRC"


def load_case_ids(path: Path, require_valid_os: bool = True) -> set[str]:
    frame = pd.read_csv(path)
    if "case_id" not in frame:
        raise ValueError(f"No case_id column in {path}")
    if require_valid_os and "survival_days" in frame:
        survival = pd.to_numeric(frame["survival_days"], errors="coerce")
        frame = frame[survival.notna() & (survival > 0)].copy()
    return {
        value for value in frame["case_id"].dropna().astype(str).str.strip().str.upper()
        if value.startswith("TCGA-") and len(value) == 12
    }


def request_with_retries(url: str, *, params=None, stream=False, timeout=600):
    last_error = None
    for attempt in range(1, 6):
        try:
            response = requests.get(
                url, params=params, stream=stream, timeout=timeout,
                headers={"User-Agent": "clinical-core/vision-preparation"},
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 5:
                break
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"TCIA request failed after 5 attempts: {last_error}")


def fetch_catalog(cache_path: Path, force: bool = False) -> pd.DataFrame:
    if cache_path.exists() and not force:
        print(f"[catalog] reusing {cache_path}", flush=True)
        return pd.read_csv(cache_path)
    print("[catalog] querying TCIA getSeries (this endpoint can take several minutes)...", flush=True)
    response = request_with_retries(
        f"{BASE_URL}/getSeries", params={"Collection": COLLECTION}
    )
    frame = pd.DataFrame(response.json())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    print(f"[catalog] {len(frame)} series saved to {cache_path}", flush=True)
    return frame


def rank_series(
    catalog: pd.DataFrame,
    case_ids: set[str],
    min_slices: int,
    max_candidates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = catalog.copy()
    frame["case_id"] = frame["PatientID"].astype(str).str.strip().str.upper()
    frame["Modality"] = frame["Modality"].fillna("").astype(str).str.upper()
    frame["ImageCount_num"] = pd.to_numeric(frame["ImageCount"], errors="coerce").fillna(0).astype(int)
    frame["FileSize_bytes"] = pd.to_numeric(frame.get("FileSize", 0), errors="coerce").fillna(0).astype("int64")
    for column in ["SeriesDescription", "StudyDesc", "BodyPartExamined"]:
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    frame["series_text"] = frame[
        ["SeriesDescription", "StudyDesc", "BodyPartExamined"]
    ].agg(" ".join, axis=1).str.lower()
    frame["abdomen_hint"] = frame["series_text"].str.contains(
        r"abd|renal|kidney|pelv|uro|neph|retroperitone", regex=True
    )
    frame["bad_hint"] = frame["series_text"].str.contains(
        r"localizer|scout|topogram|dose|report|screen|seg", regex=True
    )
    eligible = frame[
        frame["case_id"].isin(case_ids)
        & frame["Modality"].isin(["CT", "MR"])
        & (frame["ImageCount_num"] >= min_slices)
        & ~frame["bad_hint"]
    ].copy()
    eligible["modality_rank"] = eligible["Modality"].map({"CT": 0, "MR": 1}).fillna(9)
    eligible["anatomy_rank"] = (~eligible["abdomen_hint"]).astype(int)
    eligible = eligible.sort_values(
        ["case_id", "modality_rank", "anatomy_rank", "ImageCount_num"],
        ascending=[True, True, True, False],
    )
    eligible["candidate_rank"] = eligible.groupby("case_id").cumcount() + 1
    candidates = eligible[eligible["candidate_rank"] <= max_candidates].copy()
    selected = candidates[candidates["candidate_rank"] == 1].copy()
    return candidates.reset_index(drop=True), selected.reset_index(drop=True)


def safe_extract(zip_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        for member in members:
            destination = (output_dir / member.filename).resolve()
            if root not in destination.parents and destination != root:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(output_dir)
    return sum(1 for member in members if not member.is_dir())


def download_one(row: Dict, output_dir: Path, staging_dir: Path) -> Dict:
    case_id = row["case_id"]
    uid = str(row["SeriesInstanceUID"])
    series_dir = output_dir / case_id / uid
    marker = series_dir / ".complete.json"
    result = {
        "case_id": case_id,
        "SeriesInstanceUID": uid,
        "Modality": row["Modality"],
        "ImageCount_num": int(row["ImageCount_num"]),
        "FileSize_bytes": int(row["FileSize_bytes"]),
        "series_dir": str(series_dir),
    }
    series_dir.mkdir(parents=True, exist_ok=True)
    lock_path = series_dir / ".download.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if marker.exists():
            result.update({"status": "cached", "error": ""})
            return result

        staging_dir.mkdir(parents=True, exist_ok=True)
        part_path = staging_dir / f"{uid}.zip.part"
        zip_path = staging_dir / f"{uid}.zip"
        try:
            with request_with_retries(
                f"{BASE_URL}/getImage",
                params={"SeriesInstanceUID": uid, "NewFileNames": "Yes"},
                stream=True,
                timeout=1800,
            ) as response:
                with open(part_path, "wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            part_path.replace(zip_path)
            extracted_files = safe_extract(zip_path, series_dir)
            marker.write_text(json.dumps({
                "case_id": case_id,
                "series_instance_uid": uid,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "archive_bytes": zip_path.stat().st_size,
                "extracted_files": extracted_files,
            }, indent=2))
            zip_path.unlink(missing_ok=True)
            result.update({"status": "downloaded", "error": ""})
        except Exception as exc:
            result.update({"status": "failed", "error": repr(exc)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids-from", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/tcia_kirc_dicom"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests/tcia_kirc"))
    parser.add_argument("--min-slices", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--reverse", action="store_true",
        help="Process selected cases in reverse order (useful for a resumed tail pass)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Download only N patients; 0 means all")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--force-catalog", action="store_true")
    parser.add_argument(
        "--include-invalid-os", action="store_true",
        help="Include cases without positive survival_days (excluded by default)",
    )
    args = parser.parse_args()

    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.manifest_dir / "tcia_series_catalog_TCGA-KIRC.csv"
    catalog = fetch_catalog(catalog_path, force=args.force_catalog)
    case_ids = load_case_ids(
        args.case_ids_from, require_valid_os=not args.include_invalid_os
    )
    candidates, selected = rank_series(
        catalog, case_ids, args.min_slices, args.max_candidates
    )
    candidates_path = args.manifest_dir / "series_candidates_ranked.csv"
    selected_path = args.manifest_dir / "series_selected.csv"
    candidates.to_csv(candidates_path, index=False)
    selected.to_csv(selected_path, index=False)

    total_bytes = int(selected["FileSize_bytes"].sum()) if len(selected) else 0
    print(f"[selection] cohort={len(case_ids)} imaging_patients={len(selected)}", flush=True)
    print(f"[selection] estimated selected download={total_bytes / 1024**3:.2f} GiB", flush=True)
    print(f"[selection] {selected_path}", flush=True)
    if args.manifest_only:
        return 0

    if args.limit > 0:
        selected = selected.head(args.limit).copy()
    if args.reverse:
        selected = selected.iloc[::-1].reset_index(drop=True)
    records = selected.to_dict("records")
    staging = args.output_dir / ".downloads" / str(os.getpid())
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(download_one, row, args.output_dir, staging): row
            for row in records
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[download {index}/{len(records)}] {result['case_id']} "
                f"{result['Modality']} {result['status']}"
                + (f" — {result['error']}" if result["error"] else ""),
                flush=True,
            )

    result_frame = pd.DataFrame(results).sort_values("case_id")
    download_manifest = args.manifest_dir / "download_manifest.csv"
    result_frame.to_csv(download_manifest, index=False)
    provenance = {
        "collection": COLLECTION,
        "collection_doi": "10.7937/K9/TCIA.2016.V6PBVTDR",
        "license": "CC BY 3.0",
        "api": BASE_URL,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_ids_source": str(args.case_ids_from.resolve()),
        "selection": "CT first, abdomen/renal hint, largest ImageCount; localizers excluded",
        "n_selected": len(records),
        "status_counts": result_frame["status"].value_counts().to_dict(),
        "estimated_bytes": int(sum(int(row["FileSize_bytes"]) for row in records)),
    }
    (args.manifest_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    shutil.rmtree(staging, ignore_errors=True)
    print(f"[done] {download_manifest}", flush=True)
    return 1 if (result_frame["status"] == "failed").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
