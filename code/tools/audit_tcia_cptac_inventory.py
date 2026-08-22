"""Compare the current TCIA CPTAC-CCRCC inventory with MMIST mappings.

Run the MMIST/GDC survival audit first. This companion audit downloads only
TCIA series metadata, never DICOM images, and quantifies whether newer TCIA
cases increase the evaluable CPTAC survival cohort.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from audit_mmist_cptac_survival import derive_os, download, is_cptac_id  # noqa: E402


TCIA_SERIES_URL = (
    "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
    "getSeries?Collection=CPTAC-CCRCC&format=json"
)


def build_inventory_summary(
    clinical: pd.DataFrame,
    ct: pd.DataFrame,
    mr: pd.DataFrame,
    series: list[dict],
    gdc_payload: dict,
) -> tuple[pd.DataFrame, dict]:
    modalities: dict[str, set[str]] = {}
    for item in series:
        case_id = str(item.get("PatientID") or "")
        modality = str(item.get("Modality") or "")
        if case_id:
            modalities.setdefault(case_id, set()).add(modality)

    clinical_ids = {
        str(value) for value in clinical["case_id"] if is_cptac_id(value)
    }
    mmist_ct = {str(value) for value in ct["case_id"] if is_cptac_id(value)}
    mmist_mr = {str(value) for value in mr["case_id"] if is_cptac_id(value)}
    mmist_radiology = mmist_ct | mmist_mr
    by_id = {
        hit["submitter_id"]: hit
        for hit in gdc_payload.get("data", {}).get("hits", [])
        if hit.get("submitter_id")
    }

    rows = []
    for case_id in sorted(modalities):
        endpoint = derive_os(by_id[case_id]) if case_id in by_id else {
            "survival_days": None,
            "event": None,
            "endpoint_status": "not_found",
        }
        rows.append({
            "case_id": case_id,
            "has_ct_tcia": "CT" in modalities[case_id],
            "has_mr_tcia": "MR" in modalities[case_id],
            "has_rtstruct_tcia": "RTSTRUCT" in modalities[case_id],
            "in_mmist_clinical": case_id in clinical_ids,
            "in_mmist_radiology_mapping": case_id in mmist_radiology,
            **endpoint,
        })
    audit = pd.DataFrame(rows)
    known = audit[audit["in_mmist_clinical"]]
    additional = known[~known["in_mmist_radiology_mapping"]]
    extended_ct = known[known["has_ct_tcia"]]
    extended_valid = extended_ct.dropna(subset=["survival_days", "event"])
    summary = {
        "tcia_api": {
            "patients": int(len(audit)),
            "ct_patients": int(audit["has_ct_tcia"].sum()),
            "mr_patients": int(audit["has_mr_tcia"].sum()),
            "rtstruct_patients": int(audit["has_rtstruct_tcia"].sum()),
        },
        "eligibility_crosscheck": {
            "in_mmist_cptac_clinical": int(len(known)),
            "outside_mmist_cptac_clinical": int((~audit["in_mmist_clinical"]).sum()),
            "additional_within_mmist_clinical": int(len(additional)),
            "additional_with_ct": int(additional["has_ct_tcia"].sum()),
            "additional_valid_os": int(
                additional["survival_days"].notna().sum()
            ),
            "additional_events": int(additional["event"].fillna(0).sum()),
        },
        "extended_known_ccrcc_ct": {
            "mapped_cases": int(len(extended_ct)),
            "valid_os": int(len(extended_valid)),
            "events": int(extended_valid["event"].sum()),
        },
    }
    return audit, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir", type=Path, default=Path("data/external/mmist_cptac_audit")
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    required = {
        "clinical": args.work_dir / "mmist_clinical.csv",
        "ct": args.work_dir / "mmist_ct.csv",
        "mr": args.work_dir / "mmist_mr.csv",
        "gdc": args.work_dir / "gdc_cptac3_cases.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run audit_mmist_cptac_survival.py first; missing: " + ", ".join(missing)
        )
    series_path = args.work_dir / "tcia_cptac_ccrcc_series.json"
    download(TCIA_SERIES_URL, series_path, force=args.force_download)
    audit, summary = build_inventory_summary(
        pd.read_csv(required["clinical"]),
        pd.read_csv(required["ct"]),
        pd.read_csv(required["mr"]),
        json.loads(series_path.read_text()),
        json.loads(required["gdc"].read_text()),
    )
    audit.to_csv(args.work_dir / "tcia_patient_level_audit.csv", index=False)
    output = args.work_dir / "tcia_inventory_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
